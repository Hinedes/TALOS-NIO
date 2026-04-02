"""
retroactive_vrs_cleanup.py (TALOS NIO)
Instantly reclaims disk space by deleting raw ~4GB VRS directories and .zip files 
for sequences that have already been converted to an .npz cache.
"""
from pathlib import Path
import shutil

def main():
    # Paths based on TALOS standard config
    root = Path('/mnt/c/TALOS/nymeria')
    cache_dir = Path('/home/iclab/TALOS/golden/cache')
    
    print(f"Scanning {root} for obsolete VRS data...")
    print(f"Cache reference: {cache_dir}\n")
    
    if not root.exists():
        # Fallback to local Windows path if running outside WSL
        root = Path('c:/TALOS/nymeria')
        cache_dir = Path('c:/TALOS/golden/cache')
        
    if not cache_dir.exists():
        print(f"Cache directory {cache_dir} not found. Make sure sequences have been cached first.")
        return
        
    freed_bytes = 0
    cleaned_count = 0
    
    # Iterate through all cached .npz sequence files
    for cache_file in cache_dir.glob('*.npz'):
        # E.g. Nymeria_v0.0_20230607_s1_barbara_wheeler_recording_head.npz
        seq_id = cache_file.stem
        
        # 1. Target the massive 4GB extracted directory
        extract_dir = root / seq_id
        if extract_dir.exists() and extract_dir.is_dir():
            size = sum(f.stat().st_size for f in extract_dir.rglob('*') if f.is_file())
            print(f"[CLEANUP] Deleting obsolete extracted VRS dir: {extract_dir.name} ({size / 1e9:.2f} GB)")
            shutil.rmtree(extract_dir, ignore_errors=True)
            freed_bytes += size
            cleaned_count += 1
            
        # 2. Target any lingering zipped source files
        for zip_file in root.glob(f"*{seq_id}*.zip"):
            if zip_file.exists():
                size = zip_file.stat().st_size
                print(f"[CLEANUP] Deleting lingering Zip file: {zip_file.name} ({size / 1e9:.2f} GB)")
                zip_file.unlink(missing_ok=True)
                freed_bytes += size

    print(f"\n=======================================================")
    print(f"Retroactive Garbage Collection Complete!")
    print(f"Sequences Reclaimed : {cleaned_count}")
    print(f"Total Space Freed   : {freed_bytes / 1e9:.2f} GB")
    print(f"=======================================================\n")

if __name__ == '__main__':
    main()
