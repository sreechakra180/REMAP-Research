import os
import argparse
from torchvision import datasets

def download_omniglot(data_dir: str):
    """Downloads the Omniglot dataset."""
    print("Downloading Omniglot dataset...")
    datasets.Omniglot(root=data_dir, background=True, download=True)
    datasets.Omniglot(root=data_dir, background=False, download=True)
    print("Omniglot downloaded successfully.")

def download_mini_imagenet(data_dir: str):
    """Instructions for mini-ImageNet as it usually requires Kaggle or manual download."""
    print("Mini-ImageNet requires manual download due to licensing constraints.")
    print("Please download from Kaggle or other sources and extract to:")
    print(os.path.join(data_dir, "miniimagenet"))
    print("Example: kaggle datasets download -d whitemoon/miniimagenet")

def download_cifar(data_dir: str):
    """Downloads CIFAR-100 dataset."""
    print("Downloading CIFAR-100 dataset...")
    datasets.CIFAR100(root=data_dir, train=True, download=True)
    datasets.CIFAR100(root=data_dir, train=False, download=True)
    print("CIFAR-100 downloaded successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for REMAP-Net")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to store datasets")
    parser.add_argument("--dataset", type=str, choices=["all", "omniglot", "mini_imagenet", "cifar"], default="all", help="Dataset to download")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.dataset in ["all", "omniglot"]:
        download_omniglot(args.data_dir)
    
    if args.dataset in ["all", "cifar"]:
        download_cifar(args.data_dir)
        
    if args.dataset in ["all", "mini_imagenet"]:
        download_mini_imagenet(args.data_dir)
