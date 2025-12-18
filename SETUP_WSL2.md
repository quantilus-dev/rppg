# WSL2 Setup Guide for rPPG Project

This guide will help you set up the rPPG project in your WSL2 environment.

## Prerequisites

- WSL2 with Ubuntu installed (✅ You have this!)
- Access to the project files from Windows (✅ Already accessible)

## Step-by-Step Setup

### Option A: Automated Setup (Recommended)

1. **Open WSL2 Ubuntu terminal:**
   ```bash
   wsl -d Ubuntu-24.04
   ```
   Or simply open Ubuntu from your Start menu if you have it installed there.

2. **Navigate to the project directory:**
   ```bash
   cd /mnt/c/Users/theda/OneDrive/Documents/rppg
   ```

3. **Run the setup script:**
   ```bash
   chmod +x setup_rppg_wsl2.sh
   bash setup_rppg_wsl2.sh
   ```
   
   **Important:** Run this script directly in your WSL2 terminal (not via PowerShell). The script will:
   - Install Miniconda in your Linux home directory (`~/miniconda3`)
   - Create the conda environment from `rppg.yaml`
   - Install additional required packages

4. **After the script completes, activate the environment:**
   ```bash
   source ~/.bashrc  # Reload shell configuration
   conda activate rppg
   ```

5. **Install additional packages:**
   ```bash
   conda install -y wandb neurokit2 h5py scipy
   pip install self-attention-cv  # From the pip section in rppg.yaml
   ```

### Option B: Manual Setup

If you prefer to set up manually:

1. **Install Miniconda:**
   ```bash
   cd /mnt/c/Users/theda/OneDrive/Documents/rppg
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
   source $HOME/miniconda3/etc/profile.d/conda.sh
   conda init bash
   source ~/.bashrc
   ```

2. **Create the conda environment:**
   ```bash
   conda env create -f rppg.yaml
   ```

3. **Activate the environment:**
   ```bash
   conda activate rppg
   ```

4. **Install additional packages:**
   ```bash
   conda install -y wandb neurokit2 h5py scipy
   pip install self-attention-cv
   ```

## GPU Support (Optional)

If you have an NVIDIA GPU and want to use CUDA:

1. **Install NVIDIA drivers in Windows** (if not already installed)

2. **Install CUDA toolkit in WSL2:**
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
   sudo sh cuda_11.7.1_515.65.01_linux.run
   ```

   Or use conda:
   ```bash
   conda install -c nvidia cuda-toolkit=11.7
   ```

3. **Verify CUDA installation:**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Verify Installation

After setup, verify everything works:

```bash
conda activate rppg
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import numpy, scipy, cv2; print('Basic imports successful')"
```

## Usage

1. **Activate the environment:**
   ```bash
   conda activate rppg
   ```

2. **Configure the project:**
   - Edit `rppg/configs/base_config.yaml` to set your dataset paths
   - Update paths to point to your data directories

3. **Run examples:**
   ```bash
   cd /mnt/c/Users/theda/OneDrive/Documents/rppg
   python rppg/main.py
   ```

## Troubleshooting

### Conda command not found
If you get "conda: command not found", run:
```bash
source ~/.bashrc
# or
source ~/miniconda3/etc/profile.d/conda.sh
```

### Environment creation fails
If conda env create fails:
- Check internet connection
- Try updating conda: `conda update -n base -c defaults conda`
- Some packages may need to be installed manually

### CUDA issues
- Make sure NVIDIA drivers are installed in Windows
- Check WSL2 CUDA support: `nvidia-smi` (should work if drivers are installed)
- Verify PyTorch CUDA: `python -c "import torch; print(torch.version.cuda)"`

## Next Steps

1. Download and preprocess datasets (see README.md for dataset information)
2. Configure paths in `rppg/configs/base_config.yaml`
3. Run training/evaluation examples from the `examples/` directory

