# Quick Start Guide - WSL2 Setup

## Important Note
Due to environment variable conflicts when running WSL commands from PowerShell, **you need to run the setup commands directly in your WSL2 Ubuntu terminal**.

## Steps

### 1. Open WSL2 Ubuntu Terminal
You can do this in one of two ways:
- Open "Ubuntu" or "Ubuntu-24.04" from your Windows Start menu
- Or type `wsl` in PowerShell/Command Prompt

### 2. Navigate to Project Directory
```bash
cd /mnt/c/Users/theda/OneDrive/Documents/rppg
```

### 3. Make Setup Script Executable
```bash
chmod +x setup_rppg_wsl2.sh
```

### 4. Run the Setup Script
```bash
bash setup_rppg_wsl2.sh
```

This script will:
- ✅ Install Miniconda in your Linux home directory (`~/miniconda3`)
- ✅ Create the `rppg` conda environment with all dependencies
- ✅ Install additional packages (wandb, neurokit2, h5py, scipy)
- ⏱️ Takes approximately 15-30 minutes (depends on internet speed)

### 5. After Setup Completes

**Activate the environment:**
```bash
conda activate rppg
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 6. Configure the Project

Edit the configuration file to set your dataset paths:
```bash
nano rppg/configs/base_config.yaml
```

Update these paths:
- `data_root_path`: Path to your raw dataset files
- `dataset_path`: Path where preprocessed data will be saved/loaded
- `model_save_path`: Path where trained models will be saved

### 7. Start Using the Project

You can now:
- Run training examples from the `examples/` directory
- Preprocess datasets
- Train and evaluate models

Example:
```bash
cd /mnt/c/Users/theda/OneDrive/Documents/rppg
conda activate rppg
python rppg/main.py
```

## Troubleshooting

### "conda: command not found"
After installation, if conda is not found:
```bash
source ~/.bashrc
# or
source ~/miniconda3/etc/profile.d/conda.sh
```

### Environment Creation Fails
- Check your internet connection
- Try updating conda first: `conda update -n base -c defaults conda -y`
- Some packages may need manual installation

### CUDA/GPU Issues
- Ensure NVIDIA drivers are installed in Windows
- Check GPU access: `nvidia-smi` (should work from WSL2 if drivers are installed)
- Verify PyTorch CUDA: `python -c "import torch; print(torch.version.cuda)"`

## Next Steps

1. Download datasets you want to use (see README.md for available datasets)
2. Preprocess the datasets using the provided preprocessing scripts
3. Configure your training/testing parameters in the config files
4. Start training or evaluating models!

For more details, see `SETUP_WSL2.md`.

