import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing HiGATE requirements...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    # Install base requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
    
    # Install PyTorch Geometric dependencies
    torch_cuda = "cu118"  # Update based on your CUDA version
    pyg_packages = [
        f"torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+{torch_cuda}.html",
        f"torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+{torch_cuda}.html", 
        f"torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+{torch_cuda}.html",
        f"torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+{torch_cuda}.html"
    ]
    
    for package in pyg_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("Setup completed successfully!")

if __name__ == "__main__":
    install_requirements()
