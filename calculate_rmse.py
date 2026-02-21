#!/usr/bin/env python3
"""
Calculate RMSE (Root Mean Squared Error) for all models
RMSE is more sensitive to large errors than MAE
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent

print("="*80)
print("CALCULATING RMSE FOR ALL MODELS")
print("="*80)

# Load backtest results
print("\n1. Loading backtest results...")

# Check if we have saved backtest results
backtest_files = {
    'Optimized': ROOT / 'backtest_results_optimized.csv',
    'Error Analysis': ROOT / 'error_analysis_detailed.csv'
}

results = {}

for model_name, file_path in backtest_files.items():
    if file_path.exists():
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {model_name}: {len(df)} predictions")
        results[model_name] = df
    else:
        print(f"❌ {model_name} backtest file not found")

# Calculate metrics
print("\n" + "="*80)
print("METRIC CALCULATIONS")
print("="*80)

def calculate_metrics(errors):
    """Calculate MAE, RMSE, and other metrics"""
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)
    std = np.std(errors)
    return {
        'MAE': mae,
        'RMSE': rmse,
        'Bias': bias,
        'StdDev': std,
        'Min Error': np.min(errors),
        'Max Error': np.max(errors)
    }

all_metrics = {}

# Optimized model
if 'Optimized' in results:
    print("\n" + "-"*60)
    print("OPTIMIZED MODEL (Location-Specific + All Features)")
    print("-"*60)

    df = results['Optimized']
    errors = df['predicted'] - df['actual']
    metrics = calculate_metrics(errors)
    all_metrics['Optimized'] = metrics

    print(f"Total predictions: {len(df)}")
    print(f"MAE:  {metrics['MAE']:.3f} days")
    print(f"RMSE: {metrics['RMSE']:.3f} days")
    print(f"Bias: {metrics['Bias']:+.3f} days")
    print(f"StdDev: {metrics['StdDev']:.3f} days")
    print(f"Error range: [{metrics['Min Error']:.1f}, {metrics['Max Error']:.1f}] days")

    # By location
    print("\nBy location:")
    for location in df['location'].unique():
        loc_df = df[df['location'] == location]
        loc_errors = loc_df['predicted'] - loc_df['actual']
        loc_metrics = calculate_metrics(loc_errors)
        print(f"  {location:15s}: MAE={loc_metrics['MAE']:.2f}, RMSE={loc_metrics['RMSE']:.2f}, n={len(loc_df)}")

# Error Analysis (Enhanced model on all auxiliary)
if 'Error Analysis' in results:
    print("\n" + "-"*60)
    print("ENHANCED MODEL (All auxiliary data backtest)")
    print("-"*60)

    df = results['Error Analysis']
    # This has 'error' column already
    if 'error' in df.columns:
        errors = df['error'].values
    else:
        errors = df['predicted'] - df['actual']

    metrics = calculate_metrics(errors)
    all_metrics['Enhanced'] = metrics

    print(f"Total predictions: {len(df)}")
    print(f"MAE:  {metrics['MAE']:.3f} days")
    print(f"RMSE: {metrics['RMSE']:.3f} days")
    print(f"Bias: {metrics['Bias']:+.3f} days")
    print(f"StdDev: {metrics['StdDev']:.3f} days")
    print(f"Error range: [{metrics['Min Error']:.1f}, {metrics['Max Error']:.1f}] days")

    # By location
    if 'location' in df.columns:
        print("\nBy location:")
        for location in df['location'].unique():
            loc_df = df[df['location'] == location]
            if 'error' in loc_df.columns:
                loc_errors = loc_df['error'].values
            else:
                loc_errors = loc_df['predicted'] - loc_df['actual']
            loc_metrics = calculate_metrics(loc_errors)
            print(f"  {location:15s}: MAE={loc_metrics['MAE']:.2f}, RMSE={loc_metrics['RMSE']:.2f}, n={len(loc_df)}")

# Comparison
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

if len(all_metrics) > 0:
    comparison_df = pd.DataFrame(all_metrics).T
    comparison_df = comparison_df[['MAE', 'RMSE', 'Bias', 'StdDev']]
    comparison_df = comparison_df.round(3)
    print("\n" + comparison_df.to_string())

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("""
MAE (Mean Absolute Error):
  - Average of absolute errors
  - Intuitive: "On average, we're off by X days"
  - Less sensitive to outliers

RMSE (Root Mean Squared Error):
  - Square root of mean squared errors
  - MORE sensitive to large errors (penalizes outliers more)
  - RMSE > MAE means we have some large errors

Rule of thumb:
  - RMSE ≈ MAE: Errors are consistent
  - RMSE >> MAE: Some predictions are WAY off (high variance)

For our models:
""")

    for model_name, metrics in all_metrics.items():
        ratio = metrics['RMSE'] / metrics['MAE']
        print(f"\n{model_name}:")
        print(f"  RMSE/MAE ratio: {ratio:.2f}")
        if ratio < 1.2:
            print(f"  ✅ Consistent errors (low variance)")
        elif ratio < 1.5:
            print(f"  ⚠️  Moderate variance (some larger errors)")
        else:
            print(f"  🔴 High variance (many outliers/large errors)")

# Vancouver-specific analysis
print("\n" + "="*80)
print("VANCOUVER-SPECIFIC METRICS")
print("="*80)

vancouver_found = False

for model_name, df in results.items():
    if 'location' in df.columns:
        vancouver_df = df[df['location'] == 'vancouver']
        if len(vancouver_df) > 0:
            vancouver_found = True
            print(f"\n{model_name} Model:")
            print(f"  Vancouver predictions: {len(vancouver_df)}")

            if 'error' in vancouver_df.columns:
                errors = vancouver_df['error'].values
            else:
                errors = vancouver_df['predicted'] - vancouver_df['actual']

            metrics = calculate_metrics(errors)
            print(f"  MAE:  {metrics['MAE']:.2f} days")
            print(f"  RMSE: {metrics['RMSE']:.2f} days")
            print(f"  RMSE/MAE ratio: {metrics['RMSE']/metrics['MAE']:.2f}")

            if len(vancouver_df) >= 4:
                print(f"\n  Year-by-year (if available):")
                for _, row in vancouver_df.iterrows():
                    if 'year' in row:
                        actual = row['actual']
                        pred = row['predicted']
                        err = row.get('error', pred - actual)
                        print(f"    {int(row['year'])}: Actual={actual:.0f}, Pred={pred:.1f}, Error={err:+.1f}")

if not vancouver_found:
    print("\n⚠️ No Vancouver-specific data found in backtest results")
    print("   Vancouver only has 4 years of data (2022-2025)")

print("\n" + "="*80)
print("CALCULATION COMPLETE")
print("="*80)
