#!/usr/bin/env python3
"""
Master script to run all data collection and processing steps
Execute this to download and prepare all data for modeling
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Script execution order
SCRIPTS = [
    {
        'name': '01_download_openmeteo.py',
        'description': 'Download weather data from Open-Meteo API',
        'time_estimate': '1-2 minutes'
    },
    {
        'name': '02_download_climate_indices.py',
        'description': 'Download NOAA climate indices (ENSO, NAO, PDO, AO)',
        'time_estimate': '30 seconds'
    },
    {
        'name': '03_download_meteostat.py',
        'description': 'Download station weather data (optional)',
        'time_estimate': '2-5 minutes',
        'optional': True
    },
    {
        'name': '05_process_weather.py',
        'description': 'Process and aggregate weather data',
        'time_estimate': '1 minute'
    },
    {
        'name': '06_calculate_gdd.py',
        'description': 'Calculate Growing Degree Days',
        'time_estimate': '30 seconds'
    },
    {
        'name': '07_calculate_chill_hours.py',
        'description': 'Calculate chilling hours',
        'time_estimate': '30 seconds'
    },
    {
        'name': '09_create_final_dataset.py',
        'description': 'Merge all data sources into master dataset',
        'time_estimate': '30 seconds'
    }
]

def print_header():
    """Print script header"""
    print("="*80)
    print("  CHERRY BLOSSOM PREDICTION - DATA COLLECTION PIPELINE")
    print("="*80)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("  This script will:")
    print("  1. Download historical weather data (Open-Meteo + Meteostat)")
    print("  2. Download climate indices (ENSO, NAO, PDO, AO)")
    print("  3. Process and aggregate weather data")
    print("  4. Calculate Growing Degree Days (GDD)")
    print("  5. Calculate chilling hours")
    print("  6. Create final master dataset for modeling")
    print()
    print("  Total estimated time: 5-10 minutes")
    print("="*80)
    print()

def run_script(script_info, script_dir):
    """Execute a single script"""

    script_name = script_info['name']
    description = script_info['description']
    time_estimate = script_info['time_estimate']
    optional = script_info.get('optional', False)

    print()
    print("─" * 80)
    print(f"RUNNING: {script_name}")
    print(f"Task: {description}")
    print(f"Estimated time: {time_estimate}")
    if optional:
        print("Status: OPTIONAL (skipping will not break pipeline)")
    print("─" * 80)

    script_path = script_dir / script_name

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False

    try:
        # Run the script
        start_time = datetime.now()

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if result.returncode == 0:
            print()
            print(f"✅ SUCCESS - Completed in {duration:.1f} seconds")
            return True
        else:
            print()
            print(f"❌ FAILED with return code {result.returncode}")

            if optional:
                print(f"⚠️  This is optional - continuing with pipeline")
                return True
            else:
                return False

    except Exception as e:
        print(f"❌ ERROR running {script_name}: {e}")

        if optional:
            print(f"⚠️  This is optional - continuing with pipeline")
            return True
        else:
            return False

def main():
    """Main execution"""

    print_header()

    # Get script directory
    script_dir = Path(__file__).parent

    # Check Python version
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7 or higher required")
        sys.exit(1)

    # Ask for confirmation
    response = input("Ready to start data collection? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled by user")
        sys.exit(0)

    print()
    print("Starting data collection pipeline...")

    # Run each script in order
    success_count = 0
    failed_scripts = []

    for script_info in SCRIPTS:
        success = run_script(script_info, script_dir)

        if success:
            success_count += 1
        else:
            failed_scripts.append(script_info['name'])

            # Stop on critical failure (non-optional script)
            if not script_info.get('optional', False):
                print()
                print("="*80)
                print("PIPELINE STOPPED - Critical script failed")
                print(f"Failed script: {script_info['name']}")
                print("="*80)
                sys.exit(1)

    # Final summary
    print()
    print("="*80)
    print("DATA COLLECTION PIPELINE COMPLETE")
    print("="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"Scripts executed: {success_count} / {len(SCRIPTS)}")

    if failed_scripts:
        print(f"Failed (optional) scripts: {', '.join(failed_scripts)}")

    print()
    print("Next steps:")
    print("  1. Check data/final/master_dataset.csv")
    print("  2. Review data/final/data_dictionary.csv")
    print("  3. Start modeling with the master dataset")
    print()
    print("="*80)

if __name__ == '__main__':
    main()
