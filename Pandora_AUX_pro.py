#!/usr/bin/env python3
"""
Pandora AUX Processing Pipeline Driver
Processes Pandora Infrared Image (InfImg) datasets.
"""

import os
import sys
import glob
import pandora_tools

def main():
    # Default behavior: scan the designated folder
    data_dir = "/opt/data2/rowe/pandora/2026/05/30/"
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if not os.path.exists(input_path):
            print(f"Error: Path '{input_path}' not found.")
            sys.exit(1)
            
        if os.path.isdir(input_path):
            data_dir = input_path
        else:
            # Single file mode
            output_dir = "output"
            if len(sys.argv) > 2:
                output_dir = sys.argv[2]
                
            try:
                pandora_tools.run_pipeline(input_path, output_dir=output_dir)
                print("Successfully processed target file.")
                sys.exit(0)
            except Exception as e:
                print(f"Pipeline failed for '{input_path}': {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

    print(f"Scanning directory: {data_dir} for InfImg FITS files...")
    
    # Search for files with InfImg in the name
    search_pattern = os.path.join(data_dir, "*InfImg*")
    fits_files = sorted(glob.glob(search_pattern))
    
    if not fits_files:
        print(f"No InfImg files found in '{data_dir}'.")
        sys.exit(0)
        
    print(f"Found {len(fits_files)} files to process:")
    for idx, f in enumerate(fits_files):
        print(f"  [{idx + 1}] {os.path.basename(f)}")
        
    # Get date from first fits file (basename like '2026-05-30__...')
    first_basename = os.path.basename(fits_files[0])
    date_part = first_basename.split("__")[0]
    year, month, day = date_part.split("-")
    date_subdir = os.path.join(year, month, day)
    date_str = date_part.replace("-", "_")
    
    output_dir = os.path.join("output_aux_pro", date_subdir)
    print(f"\nProcessing all files grouped by target. Outputs will be saved to: {output_dir}/")
    
    # Group FITS files by target
    from astropy.io import fits
    target_to_files = {}
    for f in fits_files:
        try:
            with fits.open(f) as hdul:
                targ = hdul[0].header.get("TARG_ID", "Unknown")
            if targ not in target_to_files:
                target_to_files[targ] = []
            target_to_files[targ].append(f)
        except Exception as e:
            print(f"Warning: could not read target ID from {os.path.basename(f)}: {e}")
            
    print(f"\nFound targets:")
    for target, files in target_to_files.items():
        print(f"  - {target}: {len(files)} files")
        
    import pandas as pd
    all_dfs = []
    success_count = 0
    for target, files in target_to_files.items():
        try:
            df = pandora_tools.run_target_pipeline(files, target, output_dir=output_dir)
            if df is not None:
                all_dfs.append(df)
            success_count += len(files)
        except Exception as e:
            print(f"FAILED target pipeline for {target}: {e}")
            import traceback
            traceback.print_exc()
            
    # daily Parquet filename uses date_str computed at the start
    
    # Save daily Parquet file
    if all_dfs:
        consolidated_df = pd.concat(all_dfs, ignore_index=True)
        daily_path = os.path.join(output_dir, f"{date_str}_photometry.parquet")
        consolidated_df.to_parquet(daily_path, index=False)
        print(f"\nSaved daily time-series photometry to: {daily_path}")
        
    print(f"\nProcessing complete! Successfully processed {success_count}/{len(fits_files)} files.")

if __name__ == "__main__":
    main()
