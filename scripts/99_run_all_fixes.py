#!/usr/bin/env python3
"""
Master script to run all critical dataset fixes
Executes fixes in correct order and validates results
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Fixes to run (in order)
FIXES = [
    {
        'name': '11_fix_oni_download.py',
        'description': 'Fix ONI (ENSO) climate index download',
        'priority': 'CRITICAL',
        'time_estimate': '15 seconds'
    },
    {
        'name': '12_fix_photoperiod.py',
        'description': 'Fix photoperiod calculation (astronomical formula)',
        'priority': 'CRITICAL',
        'time_estimate': '30 seconds'
    },
    {
        'name': '10_fix_missing_competition_sites.py',
        'description': 'Download weather for Liestal, Vancouver, NYC (Meteostat)',
        'priority': 'CRITICAL',
        'time_estimate': '2-5 minutes',
        'requires_install': 'meteostat'
    }
]

# Scripts to re-run after fixes
REGENERATE = [
    {
        'name': '05_process_weather.py',
        'description': 'Reprocess weather data with new downloads'
    },
    {
        'name': '06_calculate_gdd.py',
        'description': 'Recalculate GDD with new weather data'
    },
    {
        'name': '07_calculate_chill_hours.py',
        'description': 'Recalculate chilling hours with new weather data'
    },
    {
        'name': '09_create_final_dataset.py',
        'description': 'Recreate master dataset with all fixes'
    }
]

def print_header():
    """Print script header"""
    print("="*90)
    print("  CHERRY BLOSSOM DATASET - CRITICAL FIXES")
    print("="*90)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("  This script will:")
    print("  1. Fix ONI (ENSO) climate index download (~15 sec)")
    print("  2. Fix photoperiod calculation to use proper astronomy (~30 sec)")
    print("  3. Download weather for Liestal, Vancouver, NYC (~2-5 min)")
    print("  4. Reprocess all weather data")
    print("  5. Recalculate GDD and chilling hours")
    print("  6. Recreate master dataset with all fixes")
    print()
    print("  Total estimated time: 5-10 minutes")
    print("="*90)
    print()

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")

    missing = []

    try:
        import pandas
        print("  ✓ pandas installed")
    except ImportError:
        missing.append('pandas')

    try:
        import numpy
        print("  ✓ numpy installed")
    except ImportError:
        missing.append('numpy')

    try:
        import requests
        print("  ✓ requests installed")
    except ImportError:
        missing.append('requests')

    # Meteostat is optional but recommended
    try:
        import meteostat
        print("  ✓ meteostat installed")
    except ImportError:
        print("  ⚠️  meteostat not installed (needed for fix #3)")
        print("     Install with: pip install meteostat")
        missing.append('meteostat')

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")

        if 'meteostat' in missing:
            response = input("\nContinue without meteostat? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                return False

    return True

def run_script(script_info, script_dir):
    """Execute a single script"""

    script_name = script_info['name']
    description = script_info['description']
    priority = script_info.get('priority', '')
    time_estimate = script_info.get('time_estimate', 'unknown')

    print()
    print("─" * 90)
    print(f"RUNNING: {script_name}")
    if priority:
        print(f"Priority: {priority}")
    print(f"Task: {description}")
    print(f"Estimated time: {time_estimate}")
    print("─" * 90)

    script_path = script_dir / script_name

    if not script_path.exists():
        print(f"⚠️  Script not found: {script_path}")
        return False

    try:
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
            return False

    except Exception as e:
        print(f"❌ ERROR running {script_name}: {e}")
        return False

def validate_results():
    """Validate that fixes were successful"""

    print()
    print("="*90)
    print("VALIDATING RESULTS")
    print("="*90)

    try:
        import pandas as pd

        # Check master dataset exists
        dataset_file = Path('data/final/master_dataset.csv')
        if not dataset_file.exists():
            print("❌ Master dataset not found")
            return False

        df = pd.read_csv(dataset_file)

        print(f"\n✓ Master dataset loaded: {len(df):,} records")

        # Check competition sites have weather
        comp_sites = ['kyoto', 'washingtondc', 'liestal', 'vancouver', 'newyorkcity']
        print(f"\n✓ Competition sites weather coverage:")

        all_have_weather = True
        for site in comp_sites:
            site_df = df[df['location'] == site]
            if len(site_df) > 0:
                gdd_count = site_df['GDD_jan_feb'].notna().sum()
                gdd_pct = (gdd_count / len(site_df)) * 100
                status = "✓" if gdd_count > 0 else "✗"
                print(f"  {status} {site:15s}: {gdd_count}/{len(site_df)} records ({gdd_pct:.1f}%)")

                if gdd_count == 0:
                    all_have_weather = False
            else:
                print(f"  ✗ {site:15s}: NOT FOUND")
                all_have_weather = False

        # Check climate indices
        print(f"\n✓ Climate indices coverage:")
        for idx in ['ONI_winter', 'PDO_winter', 'NAO_winter', 'AO_winter']:
            if idx in df.columns:
                count = df[idx].notna().sum()
                pct = (count / len(df)) * 100
                status = "✓" if pct > 90 else "⚠️"
                print(f"  {status} {idx:15s}: {count:,}/{len(df):,} ({pct:.1f}%)")
            else:
                print(f"  ✗ {idx:15s}: NOT IN DATASET")

        # Check photoperiod
        if 'photoperiod_mar20' in df.columns:
            photo_min = df['photoperiod_mar20'].min()
            photo_max = df['photoperiod_mar20'].max()
            # Valid range should be ~8-16 hours
            if 8 <= photo_min <= 16 and 8 <= photo_max <= 16:
                print(f"\n✓ Photoperiod: {photo_min:.2f} - {photo_max:.2f} hours (valid range)")
            else:
                print(f"\n⚠️  Photoperiod: {photo_min:.2f} - {photo_max:.2f} hours (check values)")

        if all_have_weather:
            print(f"\n✅ ALL COMPETITION SITES HAVE WEATHER DATA")
        else:
            print(f"\n⚠️  Some competition sites still missing weather data")
            print(f"   This may be due to Meteostat not finding suitable stations")

        return True

    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def main():
    """Main execution"""

    print_header()

    # Check dependencies
    if not check_dependencies():
        print("\nExiting due to missing dependencies")
        sys.exit(1)

    # Ask for confirmation
    response = input("\nReady to run all fixes? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled by user")
        sys.exit(0)

    script_dir = Path(__file__).parent

    # Phase 1: Run fixes
    print()
    print("="*90)
    print("PHASE 1: RUNNING CRITICAL FIXES")
    print("="*90)

    fix_results = []
    for fix_info in FIXES:
        success = run_script(fix_info, script_dir)
        fix_results.append((fix_info['name'], success))

    # Phase 2: Regenerate datasets
    print()
    print("="*90)
    print("PHASE 2: REGENERATING DATASETS")
    print("="*90)

    for regen_info in REGENERATE:
        run_script(regen_info, script_dir)

    # Phase 3: Validate
    validate_results()

    # Final summary
    print()
    print("="*90)
    print("FIXES COMPLETE")
    print("="*90)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Fix results:")
    for script_name, success in fix_results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {script_name}")

    print()
    print("Next steps:")
    print("  1. Check data/final/master_dataset.csv")
    print("  2. Verify all competition sites have weather data")
    print("  3. Start modeling with enhanced dataset")
    print()
    print("="*90)

if __name__ == '__main__':
    main()
