# Fix neurokit2 Python 3.9 Compatibility Issue

## Problem
The latest version of `neurokit2` uses Python 3.10+ syntax (`float | None`) which is not compatible with Python 3.9.

## Solution

Run this command in your WSL2 terminal:

```bash
cd /mnt/c/Users/theda/OneDrive/Documents/rppg
conda activate rppg
pip install 'neurokit2<0.3.0' --force-reinstall
```

Or use the fix script:

```bash
cd /mnt/c/Users/theda/OneDrive/Documents/rppg
chmod +x fix_neurokit2.sh
bash fix_neurokit2.sh
```

## Verify Fix

After running the fix, test the import:

```bash
conda activate rppg
python -c "import neurokit2 as nk; print('neurokit2 imported successfully')"
python -c "from rppg.models import get_model; print('rPPG models imported successfully')"
```

This should now work without errors!

