import os
import sys
import argparse
import urllib.request
import zipfile
import gdown


DATASETS = {
    "DIV2K_train": {
        "url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "type": "zip",
    },
    "DIV2K_val": {
        "url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        "type": "zip",
    },
    "Flickr2K": {
        "url": "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar",
        "type": "tar",
    },
    "RealSR": {
        "url": "https://drive.google.com/uc?id=1sSvEh3_3r5W3M4sX9Yp_9BfY5m1y5m1y",
        "type": "gdrive",
    },
}


def download_file(url, save_path):
    print(f"Downloading: {url}")
    if url.startswith("https://drive.google.com"):
        gdown.download(url, save_path, quiet=False)
    else:
        urllib.request.urlretrieve(url, save_path)
    print(f"Saved to: {save_path}")


def extract_archive(archive_path, extract_dir, archive_type="zip"):
    print(f"Extracting: {archive_path}")
    if archive_type == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    elif archive_type == "tar":
        import tarfile
        with tarfile.open(archive_path, "r") as tf:
            tf.extractall(extract_dir)
    print(f"Extracted to: {extract_dir}")


def prepare_dataset(dataset_name, data_dir):
    os.makedirs(data_dir, exist_ok=True)

    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available datasets: {list(DATASETS.keys())}")
        return

    info = DATASETS[dataset_name]
    save_path = os.path.join(data_dir, f"{dataset_name}.{info['type']}")

    if not os.path.exists(save_path):
        download_file(info["url"], save_path)
    else:
        print(f"Archive already exists: {save_path}")

    extract_dir = os.path.join(data_dir, dataset_name)
    if not os.path.exists(extract_dir):
        extract_archive(save_path, extract_dir, info["type"])
    else:
        print(f"Dataset already extracted: {extract_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for CISR training")
    parser.add_argument("--dataset", type=str, default="all",
                        help="Dataset name or 'all' for all datasets")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory to store datasets")
    args = parser.parse_args()

    if args.dataset == "all":
        for name in DATASETS:
            prepare_dataset(name, args.data_dir)
    else:
        prepare_dataset(args.dataset, args.data_dir)


if __name__ == "__main__":
    main()
