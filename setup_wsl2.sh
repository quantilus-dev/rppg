#!/bin/bash
# Setup script for rPPG project in WSL2
# This script installs Miniconda and sets up the rPPG environment

set -e  # Exit on error

echo "=========================================="
echo "rPPG Project WSL2 Setup Script"
echo "=========================================="

# Check if we're in WSL2
if [ ! -d "/mnt/c" ]; then
    echo "Error: This script should be run in WSL2"
    exit 1
fi

# Navigate to project directory
PROJECT_DIR="/mnt/c/Users/theda/OneDrive/Documents/rppg"
cd "$PROJECT_DIR"
echo "Project directory: $PROJECT_DIR"

# Check if conda is already installed
if command -v conda &> /dev/null; then
    echo "Conda is already installed!"
    conda --version
else
    echo "Installing Miniconda..."
    
    # Download Miniconda installer
    MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
    if [ ! -f "$MINICONDA_INSTALLER" ]; then
        echo "Downloading Miniconda..."
        wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER
    fi
    
    # Install Miniconda
    echo "Installing Miniconda (this may take a few minutes)..."
    bash $MINICONDA_INSTALLER -b -p $HOME/miniconda3
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init bash
    
    # Add conda to PATH for current session
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    echo "Miniconda installed successfully!"
    conda --version
fi

# Initialize conda for bash (if not already done)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# Verify conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not available. Please restart your terminal or run:"
    echo "  source ~/.bashrc"
    exit 1
fi

echo ""
echo "Creating conda environment from rppg.yaml..."
echo "This will take several minutes (downloading and installing packages)..."

# Update conda first
conda update -n base -c defaults conda -y

# Create environment from yaml file
conda env create -f rppg.yaml

echo ""
echo "=========================================="
echo "Environment created successfully!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate rppg"
echo ""
echo "Additional packages that may be needed (install after activating):"
echo "  conda activate rppg"
echo "  conda install -y wandb neurokit2 h5py scipy"
echo ""
echo "Note: You may need to install CUDA toolkit separately if you want GPU support."
echo "For CUDA 11.7 (required by the environment), check NVIDIA's website."

