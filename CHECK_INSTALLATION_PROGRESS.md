# How to Check Installation Progress

## While the Setup Script is Running

### What You Should See

When running `bash setup_rppg_wsl2.sh`, you'll see several phases:

#### Phase 1: Miniconda Installation (if not already installed)
- **Download progress**: Shows percentage and download speed
  ```
  Downloading Miniconda... 
  50% [=========>        ] 50MB/s
  ```
- **Installation messages**: Various conda initialization messages
- **Completion**: "✓ Miniconda installed successfully!"

#### Phase 2: Conda Environment Creation (Longest Step - 15-30 minutes)
- **Progress indicators**:
  ```
  [2/3] Creating conda environment 'rppg' from rppg.yaml...
  ```
  
- **What you'll see during installation**:
  - Package download progress bars:
    ```
    pytorch-2.0.1    | #################################### | 100%
    numpy-1.24.3     | #################################### | 100%
    ```
  - Package extraction messages:
    ```
    Extracting packages...
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    ```
  - Installation of each package with messages like:
    ```
    Installing pytorch...
    Installing numpy...
    Linking packages...
    ```

#### Phase 3: Additional Packages
- Installing wandb, neurokit2, h5py, scipy
- Installing pip packages (self-attention-cv)

### Signs Installation is Working

✅ **Good signs:**
- You see download progress percentages
- Packages are being extracted ("Extracting...", "Linking...")
- No error messages
- The script continues without hanging

⚠️ **Normal but concerning (usually fine):**
- Script appears to "hang" - it's actually downloading/installing
- Warnings about packages (these are usually harmless)
- "Solving environment" taking a long time (can take 5-10 minutes)

❌ **Bad signs (if you see these, something is wrong):**
- Error messages that say "FAILED" or "ERROR"
- Script exits unexpectedly
- "No space left on device" errors

## How to Monitor Progress in Another Terminal

If you want to check progress from another terminal window:

### 1. Check if the Script is Still Running
```bash
# In a new WSL2 terminal, check running processes
ps aux | grep setup_rppg_wsl2.sh
```

### 2. Check Conda Environment Status
```bash
# See if environment is being created
conda env list

# You should see 'rppg' listed (even if incomplete)
```

### 3. Check Downloaded Packages
```bash
# See conda's package cache (shows what's been downloaded)
ls -lh ~/.conda/pkgs/ | tail -20

# Or if using miniconda
ls -lh ~/miniconda3/pkgs/ | tail -20
```

### 4. Check Disk Space (Important!)
```bash
# Make sure you have enough space (need at least 10GB free)
df -h ~
```

### 5. Monitor Network Activity
```bash
# Check if downloads are happening (in another terminal)
# You should see network activity if packages are downloading
iftop  # if installed
# Or
nethogs  # if installed
```

## If Installation Seems Stuck

### Wait First
- Environment creation can take 15-30 minutes
- "Solving environment" step alone can take 5-10 minutes
- Large packages (like PyTorch ~2GB) take time to download

### Check if It's Actually Working
```bash
# In another terminal, check conda processes
ps aux | grep conda

# Check network activity (should show downloads if working)
# On Windows, check Task Manager -> Performance -> Ethernet
```

### If It's Been Over 30 Minutes with No Progress

1. **Check the terminal output** - Look for any error messages
2. **Check disk space**:
   ```bash
   df -h ~
   ```
   Need at least 10GB free
3. **Check internet connection**:
   ```bash
   ping -c 3 8.8.8.8
   ```
4. **Try interrupting (Ctrl+C) and check logs**:
   ```bash
   # Check conda logs
   cat ~/.conda/pkgs/cache/*.json 2>/dev/null | tail -50
   ```

## After Installation Completes

### Verify Installation Success

1. **Check environment exists:**
   ```bash
   conda env list
   # Should show 'rppg' with an asterisk (*) if active
   ```

2. **Activate and test:**
   ```bash
   conda activate rppg
   python --version  # Should show Python 3.9.x
   python -c "import torch; print(torch.__version__)"  # Should print version
   python -c "import numpy; print(numpy.__version__)"  # Should print version
   ```

3. **Check key packages:**
   ```bash
   conda activate rppg
   python -c "import torch, numpy, cv2, scipy; print('All imports successful!')"
   ```

4. **Check CUDA (if you have GPU):**
   ```bash
   conda activate rppg
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 
   ```

## Common Issues and Solutions

### "Solving environment" taking forever
- This is normal, can take 5-15 minutes
- Be patient, conda is finding compatible package versions

### Download speeds are slow
- Normal for large packages (PyTorch is ~2GB)
- You can monitor progress - as long as it's downloading, it's working

### Installation failed partway through
- Check error messages in the terminal
- Try running the script again (conda will resume from where it stopped)
- Or manually create environment: `conda env create -f rppg.yaml`

### "No space left on device"
- Free up disk space (need ~10GB)
- Or install to a different location with more space

## Quick Status Check Commands

Create a quick status checker script:
```bash
# Save this as check_install.sh
#!/bin/bash
echo "=== Installation Status Check ==="
echo ""
echo "Conda installed:"
command -v conda &> /dev/null && echo "✓ Yes - $(conda --version)" || echo "✗ No"
echo ""
echo "rppg environment:"
conda env list | grep rppg && echo "✓ Exists" || echo "✗ Not found"
echo ""
echo "Disk space:"
df -h ~ | tail -1
echo ""
echo "If environment exists, test it:"
if conda env list | grep -q rppg; then
    eval "$(conda shell.bash hook)"
    conda activate rppg 2>/dev/null && python -c "import torch; print('✓ PyTorch:', torch.__version__)" || echo "✗ Cannot activate/test"
fi
```

Run it with:
```bash
chmod +x check_install.sh
bash check_install.sh
```

