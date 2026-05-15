# CISR: Causal Intervention for Blind Image Super-Resolution



PyTorch implementation of **CISR** — reformulates blind SR as causal intervention problem via Structural Causal Model (SCM).

## Algorithm

**Core Insight**: LR image $X$ is a collider of independent content $C$ and degradation $D$. Existing methods capture spurious correlations; CISR implements do-calculus to decouple them.

**Architecture**:
- **Content Encoder $E_c$**: DINOv2-ViT-S/14 backbone, multi-scale aggregation {L/4, L/2, 3L/4, L}, 256-dim latent
- **Degradation Encoder $E_d$**: 5-layer strided CNN + GAP, GroupNorm, 256-dim descriptor
- **CAFI Module**: Cross-attention with causal gates: $z_{fused} = z_c + s \cdot G(Attn(z_c, z_d))$, $s = \sigma(W_s[z_c; z_d])$
- **Decoder $G_\theta$**: 16 ResBlocks + AdaIN modulation from $z_d$
- **Clean Prior $z_{clean}$**: Learnable 256-dim vector, init=0, L2 regularized

**Intervention via Contrastive Learning**:
- Content positives: same $c$, different $d$ → simulates $do(C=c)$
- Degradation positives: same $d$, different $c$ → simulates $do(D=d)$
- InfoNCE loss with $\tau=0.07$, $K=1024$ negatives

**Loss**: $\mathcal{L}_{total} = \lambda_{rec}\mathcal{L}_{rec} + \lambda_{SR}\mathcal{L}_{SR} + \lambda_{KL}\mathcal{L}_{KL} + \lambda_{NCE}(\mathcal{L}_{NCE}^c + \mathcal{L}_{NCE}^d) + \lambda_{adv}\mathcal{L}_{adv} + \lambda_{prior}\mathcal{L}_{prior}$



**Inference**: Single forward pass with $z_{clean}$ replacing $z_d$ (counterfactual)

## Project Structure

```
CISR/
├── src/
│   ├── models/          # cisr_model, encoders, cafi, decoder, discriminator
│   ├── data/            # degradation pipeline, dataset
│   ├── losses/          # all loss functions
│   ├── train.py         # training script
│   ├── inference.py     # inference script
│   └── evaluate.py      # evaluation script
├── configs/default.yaml
├── data/prepare_data.py
├── tests/test_modules.py
├── requirements.txt
├── README.md
└── LICENSE
```

## Quick Start

```bash
# Environment
conda create -n cisr python=3.10 -y
conda activate cisr
pip install -r requirements.txt

# Training (8 GPUs)
torchrun --nproc_per_node=8 -m src.train --config configs/default.yaml --output_dir results/train

# Inference
python -m src.inference --input lr.png --output results/sr --checkpoint results/train/checkpoint_final.pth

# Evaluation
python -m src.evaluate --checkpoint results/train/checkpoint_final.pth --lr_dir data/val/LR --hr_dir data/val/HR --output_dir results/eval
```

## Training Config

- Optimizer: Adam ($\beta_1=0.9$, $\beta_2=0.999$), LR=$2\times10^{-4}$, Cosine Annealing
- Iterations: 500,000 | Batch: 8 | HR patch: 192×192 (×4 SR)
- Degradation: Real-ESRGAN style (blur, noise, JPEG, 2nd-order with p=0.5)
- Adaptive triplet strategy, $N_d=8$ variants per image

## Results

| Dataset | Method | PSNR↑ | LPIPS↓ | CLIP↑ |
|---|---|---|---|---|
| RealSR | CISR (Ours) | **26.38** | **0.2615** | **0.7688** |
| | SeeSR | 25.24 | 0.3007 | 0.6700 |
| DPED | CISR (Ours) | **22.51** | **0.3608** | **0.7245** |
| | PASD | 22.04 | 0.4289 | 0.4856 |




