import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install HiGATE requirements"""
    print("Installing HiGATE requirements...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    # Install base requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
    
    # Install PyTorch Geometric with CUDA support
    try:
        import torch
        cuda_version = torch.version.cuda
        if cuda_version:
            torch_cuda = f"cu{cuda_version.replace('.', '')}"
        else:
            torch_cuda = "cpu"
    except:
        torch_cuda = "cpu"
    
    pyg_packages = [
        f"torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+{torch_cuda}.html",
        f"torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+{torch_cuda}.html", 
        f"torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+{torch_cuda}.html",
        f"torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+{torch_cuda}.html"
    ]
    
    for package in pyg_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}, trying without CUDA...")
            # Try CPU version
            package_cpu = package.split(' ')[0]
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_cpu])
    
    print("HiGATE environment setup completed!")

def verify_installation():
    """Verify that all required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'torch_geometric', 'numpy', 'scikit-learn',
        'scipy', 'pandas', 'matplotlib', 'seaborn', 'tqdm', 'PIL',
        'opencv-python', 'albumentations', 'transformers'
    ]
    
    print("Verifying installation...")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
    
    print("\nInstallation verification completed!")

if __name__ == "__main__":
    install_requirements()
    verify_installation()
