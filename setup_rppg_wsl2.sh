#!/bin/bash
# Setup script for rPPG project in WSL2
# Run this script from within WSL2 Ubuntu terminal
# Usage: bash setup_rppg_wsl2.sh

set -e  # Exit on error

echo "=========================================="
echo "rPPG Project WSL2 Setup Script"
echo "=========================================="
echo ""

# Check if we're in WSL2
if [ ! -d "/mnt/c" ]; then
    echo "Error: This script should be run in WSL2"
    exit 1
fi

# Ensure we're using Linux home directory
export HOME=$(eval echo ~$USER)
echo "Using HOME directory: $HOME"

# Project directory (on Windows filesystem)
PROJECT_DIR="/mnt/c/Users/theda/OneDrive/Documents/rppg"
cd "$PROJECT_DIR"
echo "Project directory: $PROJECT_DIR"
echo ""

# Check if conda is already installed
if command -v conda &> /dev/null; then
    echo "✓ Conda is already installed!"
    conda --version
    CONDA_BASE=$(conda info --base)
    echo "Conda base: $CONDA_BASE"
else
    echo "Installing Miniconda..."
    echo ""
    
    # Download Miniconda installer to Linux filesystem
    INSTALLER_DIR="$HOME"
    cd "$INSTALLER_DIR"
    
    MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
    if [ ! -f "$MINICONDA_INSTALLER" ]; then
        echo "Downloading Miniconda (this may take a few minutes)..."
        wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER
    fi
    
    # Install Miniconda in Linux filesystem
    echo "Installing Miniconda to $HOME/miniconda3..."
    echo "This will take several minutes..."
    bash $MINICONDA_INSTALLER -b -p "$HOME/miniconda3"
    
    # Initialize conda
    "$HOME/miniconda3/bin/conda" init bash
    
    # Add conda to PATH for current session
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    echo ""
    echo "✓ Miniconda installed successfully!"
    "$HOME/miniconda3/bin/conda" --version
    echo ""
    echo "NOTE: You may need to restart your terminal or run: source ~/.bashrc"
fi

# Source conda for bash
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# Verify conda is available
if ! command -v conda &> /dev/null; then
    echo ""
    echo "Error: Conda is not available. Please restart your terminal or run:"
    echo "  source ~/.bashrc"
    echo "Then run this script again."
    exit 1
fi

# Navigate back to project directory
cd "$PROJECT_DIR"

echo ""
echo "Checking if rppg environment already exists..."
if conda env list | grep -q "^rppg "; then
    echo "✓ rppg environment already exists!"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing rppg environment..."
        conda env remove -n rppg -y
    else
        echo "Using existing environment. To activate it, run:"
        echo "  conda activate rppg"
        exit 0
    fi
fi

echo ""
echo "Creating conda environment from rppg.yaml..."
echo "This will take 15-30 minutes (downloading and installing packages)..."
echo "Progress indicators:"
echo "  - You'll see package download progress (percentage and speed)"
echo "  - Package extraction and installation messages"
echo "  - This is normal - just wait for completion"
echo ""

# Accept Conda Terms of Service (required for Anaconda channels)
echo "[1/3] Accepting Conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Update conda first (with progress output)
echo ""
echo "[2/4] Updating conda..."
conda update -n base -c defaults conda -y

# Create environment from yaml file (verbose to show progress)
echo ""
echo "[3/4] Creating conda environment 'rppg' from rppg.yaml..."
echo "This step downloads and installs ~200 packages - be patient!"
echo "You can monitor progress below:"
echo ""
conda env create -f rppg.yaml --verbose

echo ""
echo "=========================================="
echo "✓ Environment created successfully!"
echo "=========================================="
echo ""
echo "[4/4] Installing additional packages..."
conda activate rppg

# Install packages available via conda
echo "Installing conda packages (wandb, h5py, scipy)..."
conda install -y wandb h5py scipy

# Install packages available via pip (neurokit2 is not in conda channels)
# Pin NumPy to <2.0 for compatibility, and use Python 3.9 compatible neurokit2
echo "Installing pip packages (neurokit2, self-attention-cv)..."
pip install 'numpy<2.0' 'neurokit2==0.2.12' self-attention-cv

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To use the environment:"
echo "  1. Activate it: conda activate rppg"
echo "  2. Navigate to project: cd $PROJECT_DIR"
echo "  3. Run examples or configure as needed"
echo ""
echo "Note: If you have an NVIDIA GPU, you may want to verify CUDA support:"
echo "  python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\""

