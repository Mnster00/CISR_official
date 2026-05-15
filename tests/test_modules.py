import unittest
import torch
import torch.nn as nn
import numpy as np

from src.models.content_encoder import ContentEncoder
from src.models.degradation_encoder import DegradationEncoder
from src.models.cafi_module import CAFIModule
from src.models.decoder import CausalGatedDecoder, AdaINModulation, ResBlockWithAdaIN
from src.models.discriminator import UNetDiscriminator
from src.losses import (
    ReconstructionLoss,
    KLDivergenceLoss,
    InfoNCELoss,
    PriorRegularization,
    HingeAdversarialLoss,
    CISRLoss,
)
from src.data.degradation import DegradationPipeline


class TestDegradationEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = DegradationEncoder(in_channels=3, latent_dim=256)
        self.batch_size = 2
        self.x = torch.randn(self.batch_size, 3, 48, 48)

    def test_forward_shape(self):
        mu, log_var = self.encoder(self.x)
        self.assertEqual(mu.shape, (self.batch_size, 256))
        self.assertEqual(log_var.shape, (self.batch_size, 256))

    def test_sample_deterministic(self):
        mu, log_var = self.encoder(self.x)
        z = self.encoder.sample(mu, log_var, deterministic=True)
        self.assertTrue(torch.allclose(z, mu))

    def test_sample_stochastic(self):
        mu, log_var = self.encoder(self.x)
        z = self.encoder.sample(mu, log_var, deterministic=False)
        self.assertEqual(z.shape, mu.shape)


class TestCAFIModule(unittest.TestCase):
    def setUp(self):
        self.module = CAFIModule(
            content_dim=256, degradation_dim=256, num_heads=8, dropout=0.0
        )
        self.batch_size = 2
        self.z_c = torch.randn(self.batch_size, 256)
        self.z_d = torch.randn(self.batch_size, 256)

    def test_forward_shape(self):
        z_fused = self.module(self.z_c, self.z_d)
        self.assertEqual(z_fused.shape, (self.batch_size, 256))

    def test_causal_strength_range(self):
        s = self.module.causal_strength(
            torch.cat([self.z_c, self.z_d], dim=-1)
        )
        self.assertTrue((s >= 0).all())
        self.assertTrue((s <= 1).all())


class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.decoder = CausalGatedDecoder(
            latent_dim=256, num_features=64, num_res_blocks=4,
            degradation_dim=256, scale_factor=4, out_channels=3,
        )
        self.batch_size = 2
        self.z_fused = torch.randn(self.batch_size, 256)
        self.z_d = torch.randn(self.batch_size, 256)

    def test_forward_shape(self):
        output = self.decoder(self.z_fused, self.z_d, input_size=(12, 12))
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], 3)
        self.assertEqual(output.shape[2], 12 * 4)
        self.assertEqual(output.shape[3], 12 * 4)


class TestDiscriminator(unittest.TestCase):
    def setUp(self):
        self.disc = UNetDiscriminator(in_channels=3, base_channels=32, num_layers=5)
        self.batch_size = 2
        self.x = torch.randn(self.batch_size, 3, 256, 256)

    def test_forward_shape(self):
        out = self.disc(self.x)
        self.assertEqual(out.shape[0], self.batch_size)
        self.assertEqual(out.shape[1], 1)


class TestReconstructionLoss(unittest.TestCase):
    def setUp(self):
        self.loss_l1 = ReconstructionLoss("l1")
        self.loss_l2 = ReconstructionLoss("l2")
        self.pred = torch.randn(2, 3, 48, 48)
        self.target = torch.randn(2, 3, 48, 48)

    def test_l1_loss(self):
        loss = self.loss_l1(self.pred, self.target)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss.item() >= 0)

    def test_l2_loss(self):
        loss = self.loss_l2(self.pred, self.target)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss.item() >= 0)

    def test_zero_loss(self):
        loss = self.loss_l1(self.pred, self.pred)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)


class TestKLDivergenceLoss(unittest.TestCase):
    def setUp(self):
        self.loss = KLDivergenceLoss()
        self.mu = torch.zeros(4, 256)
        self.log_var = torch.zeros(4, 256)

    def test_zero_kl(self):
        loss = self.loss(self.mu, self.log_var)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_nonzero_kl(self):
        mu = torch.ones(4, 256)
        log_var = torch.ones(4, 256)
        loss = self.loss(mu, log_var)
        self.assertTrue(loss.item() > 0)


class TestInfoNCELoss(unittest.TestCase):
    def setUp(self):
        self.loss = InfoNCELoss(temperature=0.07)
        self.z_ref = torch.randn(4, 256)
        self.z_pos = self.z_ref + 0.01 * torch.randn(4, 256)
        self.z_neg = torch.randn(8, 256)

    def test_forward(self):
        loss = self.loss(self.z_ref, self.z_pos, self.z_neg)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss.item() >= 0)


class TestPriorRegularization(unittest.TestCase):
    def setUp(self):
        self.loss = PriorRegularization()
        self.z_clean = torch.zeros(256)

    def test_zero_prior(self):
        loss = self.loss(self.z_clean)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_nonzero_prior(self):
        z = torch.ones(256)
        loss = self.loss(z)
        self.assertAlmostEqual(loss.item(), 256.0, places=3)


class TestHingeAdversarialLoss(unittest.TestCase):
    def setUp(self):
        self.loss = HingeAdversarialLoss()

    def test_discriminator_loss(self):
        d_real = torch.randn(2, 1, 8, 8)
        d_fake = torch.randn(2, 1, 8, 8)
        loss = self.loss.discriminator_loss(d_real, d_fake)
        self.assertTrue(loss.item() >= 0)

    def test_generator_loss(self):
        d_fake = torch.randn(2, 1, 8, 8)
        loss = self.loss.generator_loss(d_fake)
        self.assertEqual(loss.shape, ())


class TestCISRLoss(unittest.TestCase):
    def setUp(self):
        self.loss = CISRLoss(
            lambda_rec=1.0, lambda_sr=1.0, lambda_kl=0.01,
            lambda_nce=0.1, lambda_adv=0.1, lambda_prior=0.001,
            temperature=0.07,
        )

    def test_initialization(self):
        self.assertEqual(self.loss.lambda_rec, 1.0)
        self.assertEqual(self.loss.lambda_kl, 0.01)
        self.assertEqual(self.loss.lambda_nce, 0.1)


class TestDegradationPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = DegradationPipeline(scale_factor=4)
        self.hr_image = torch.rand(3, 192, 192)

    def test_output_shape(self):
        lr_image = self.pipeline(self.hr_image)
        self.assertEqual(lr_image.shape[0], 3)

    def test_sample_degradation_params(self):
        params = self.pipeline.sample_degradation_params()
        self.assertIn("blur_kernel", params)
        self.assertIn("noise_level", params)
        self.assertIn("jpeg_quality", params)

    def test_specific_degradation(self):
        params = self.pipeline.sample_degradation_params()
        lr_image = self.pipeline.apply_specific_degradation(self.hr_image, params)
        self.assertEqual(lr_image.shape[0], 3)


class TestAdaINModulation(unittest.TestCase):
    def setUp(self):
        self.adain = AdaINModulation(num_features=64, degradation_dim=256)
        self.h = torch.randn(2, 64, 12, 12)
        self.z_d = torch.randn(2, 256)

    def test_forward_shape(self):
        out = self.adain(self.h, self.z_d)
        self.assertEqual(out.shape, self.h.shape)


if __name__ == "__main__":
    unittest.main()
