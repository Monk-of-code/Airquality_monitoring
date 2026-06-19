import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# --- PATH FIX ---
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Now these imports will work if the files are in the same folder
try:
    from load import get_conn, ensure_bronze_tables, load_raw_csv, load_calibrated_csv
    from classifier import clean_and_write
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Check if load.py and classifier.py are in: {current_dir}")
    sys.exit(1)

# Define your data source correctly
DATA_ROOT = Path("/mnt/e/Airquality/Datasets")
BASE_DIR = DATA_ROOT / "RAW" / "CALIBRATEDPM2.5"

def main() -> None:
    # Load environment variables from the project root
    env_path = Path("/home/wanji/projects/airquality/.env")
    load_dotenv(env_path)
    
    print(f"Starting Global Ingestion from: {BASE_DIR}")

    # 1. Initialize Connection
    try:
        conn = get_conn()
        ensure_bronze_tables(conn)
    except Exception as e:
        print(f"Could not connect to database: {e}")
        return

    # 2. Iterate through Month Folders
    if not BASE_DIR.exists():
        print(f"Error: Directory {BASE_DIR} does not exist.")
        return

    month_folders = sorted([p for p in BASE_DIR.iterdir() if p.is_dir() and not p.name.startswith('_')])
    
    for month_dir in month_folders:
        print(f"\n--- MONTH: {month_dir.name} ---")
        
        csv_files = sorted(month_dir.glob("*.csv"))
        if not csv_files:
            print(f"    [SKIP] No CSV files found in {month_dir.name}")
            continue
            
        clean_root = month_dir / "_cleaned_data"

        for csv_file in csv_files:
            try:
                # Clean the file[cite: 1]
                clean_fp, level, rows = clean_and_write(csv_file, clean_root)
                
                # Load the file based on the detected level[cite: 1, 2]
                if level == "raw":
                    load_raw_csv(conn, clean_fp)
                else:
                    # Passes filename for metadata[cite: 2]
                    load_calibrated_csv(conn, clean_fp, source_file=csv_file.name)
                
                # Commit successful ingestion
                conn.commit()
                print(f"    [OK] {csv_file.name} ({rows} rows)")

            except Exception as e:
                # Correctly aligned except block
                conn.rollback()
                print(f"    [FAIL] {csv_file.name}: {e}")
                print("    Skipping to next file...")

    conn.close()
    print("\nData Ingestion Complete.")

if __name__ == "__main__":
    main()