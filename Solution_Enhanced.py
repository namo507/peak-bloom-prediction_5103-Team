#!/usr/bin/env python3
"""
Cherry Blossom Peak Bloom Prediction 2026 — Enhanced Python Pipeline

This enhanced version uses a comprehensive master dataset with:
- Growing Degree Days (GDD) and chilling hours
- Climate indices (ONI, NAO, PDO, AO)
- Hopkins' Bioclimatic Index
- Accurate photoperiod calculations
- Piecewise year splines
- Interaction terms

Target: Improve MAE from baseline ~5.6 days to 4.0-4.5 days
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

np.random.seed(5103)

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path('.')
MASTER_DATASET = ROOT / 'data/final/master_dataset.csv'

# Competition sites for 2026 prediction
COMPETITION_SITES = ['kyoto', 'washingtondc', 'liestal', 'vancouver', 'newyorkcity']

# ============================================================================
# DATA LOADING
# ============================================================================

print("="*80)
print("LOADING DATA")
print("="*80)

# Load original bloom data files (for proper source attribution)
competition_files = [
    ROOT / 'data/kyoto.csv',
    ROOT / 'data/washingtondc.csv',
    ROOT / 'data/liestal.csv',
    ROOT / 'data/vancouver.csv',
    ROOT / 'data/nyc.csv',
]

aux_files = [
    ROOT / 'data/japan.csv',
    ROOT / 'data/meteoswiss.csv',
    ROOT / 'data/south_korea.csv',
]

def read_bloom_file(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({
        'source': source,
        'location': df['location'].astype(str),
        'lat': pd.to_numeric(df['lat'], errors='coerce'),
        'long': pd.to_numeric(df['long'], errors='coerce'),
        'alt': pd.to_numeric(df['alt'], errors='coerce'),
        'year': pd.to_numeric(df['year'], errors='coerce').astype('Int64'),
        'bloom_doy': pd.to_numeric(df['bloom_doy'], errors='coerce')
    })

competition_raw = pd.concat([read_bloom_file(p, 'competition') for p in competition_files], ignore_index=True)
aux_raw = pd.concat([read_bloom_file(p, 'auxiliary') for p in aux_files], ignore_index=True)

# NYC enrichment from USA-NPN (same as original Solution.ipynb)
npn = pd.read_csv(ROOT / 'data/USA-NPN_status_intensity_observations_data.csv')
npn = npn[(npn['Site_ID'] == 32789) & (npn['Species_ID'] == 228) & (npn['Phenophase_ID'] == 501)].copy()
npn['Observation_Date'] = pd.to_datetime(npn['Observation_Date'], format='%m/%d/%y', errors='coerce')
npn['year'] = npn['Observation_Date'].dt.year
npn_yes = (
    npn[npn['Phenophase_Status'] == 1]
    .sort_values('Observation_Date')
    .groupby('year', as_index=False)
    .first()
)

nyc_npn_status = pd.DataFrame({
    'source': 'npn',
    'location': 'newyorkcity',
    'lat': 40.73040,
    'long': -73.99809,
    'alt': 8.5,
    'year': npn_yes['year'].astype('Int64'),
    'bloom_doy': pd.to_numeric(npn_yes['Day_of_Year'], errors='coerce')
}).dropna(subset=['year', 'bloom_doy'])

# USA-NPN individual phenometrics
pheno = pd.read_csv(ROOT / 'data/USA-NPN_individual_phenometrics_data.csv')
pheno = pheno[(pheno['Site_ID'] == 32789) & (pheno['Species_ID'] == 228) & (pheno['Phenophase_ID'] == 501)].copy()
nyc_npn_pheno = (
    pheno.groupby('First_Yes_Year', as_index=False)['First_Yes_DOY']
    .min()
    .rename(columns={'First_Yes_Year': 'year', 'First_Yes_DOY': 'bloom_doy'})
)
nyc_npn_pheno = nyc_npn_pheno[nyc_npn_pheno['bloom_doy'].notna()].copy()
nyc_npn_pheno['source'] = 'npn'
nyc_npn_pheno['location'] = 'newyorkcity'
nyc_npn_pheno['lat'] = 40.73040
nyc_npn_pheno['long'] = -73.99809
nyc_npn_pheno['alt'] = 8.5
nyc_npn_pheno['year'] = nyc_npn_pheno['year'].astype('Int64')

# Merge NPN sources
status_years = set(nyc_npn_status['year'].dropna().astype(int))
nyc_npn_pheno_new = nyc_npn_pheno[~nyc_npn_pheno['year'].astype(int).isin(status_years)]
nyc_npn = pd.concat([nyc_npn_status, nyc_npn_pheno_new], ignore_index=True)

existing_nyc_years = set(competition_raw.loc[competition_raw['location'] == 'newyorkcity', 'year'].dropna().astype(int))
nyc_npn = nyc_npn[~nyc_npn['year'].astype(int).isin(existing_nyc_years)]

competition = pd.concat([competition_raw, nyc_npn], ignore_index=True)

# Load master dataset for enhanced features
master_features = pd.read_csv(MASTER_DATASET)

print(f"\nLoaded bloom data:")
print(f"  Competition records: {len(competition):,}")
print(f"  Auxiliary records: {len(aux_raw):,}")

print(f"\nLoaded master dataset features:")
print(f"  Total records: {len(master_features):,}")
print(f"  Features: {len(master_features.columns)}")

# Combine bloom data with master dataset features
all_data = (
    pd.concat([competition, aux_raw], ignore_index=True)
    .dropna(subset=['year', 'bloom_doy'])
    .query('year >= 1880')
    .copy()
)
all_data['year'] = all_data['year'].astype(int)
all_data['site_id'] = all_data['source'] + '::' + all_data['location']

# Merge with enhanced features from master dataset
# Match on location and year to get weather/climate features
all_data = all_data.merge(
    master_features[['location', 'year', 'T_mean_winter', 'T_mean_spring', 'T_mean_jan_feb',
                     'GDD_jan', 'GDD_feb', 'GDD_jan_feb', 'GDD_winter', 'GDD_nov_dec',
                     'chill_hours_winter', 'chill_hours_nov_dec', 'chill_hours_nov_feb',
                     'chill_hours_dec', 'chill_hours_jan',
                     'ONI_winter', 'ONI_spring', 'NAO_winter', 'PDO_annual', 'AO_winter',
                     'Hopkins_Index', 'photoperiod_mar20', 'alt_log1p',
                     'year_c', 'year_seg_1950', 'year_seg_1980', 'year_seg_2000',
                     'lat_x_GDD', 'lat_x_chill', 'year_c_x_GDD']],
    on=['location', 'year'],
    how='left'
)

print(f"\nMerged dataset:")
print(f"  Total records: {len(all_data):,}")
print(f"  Sources: {all_data['source'].value_counts().to_dict()}")
print(f"  Locations: {all_data['location'].nunique()}")

# Target year for prediction
target_year = int(competition['year'].max()) + 1
print(f"\nTarget prediction year: {target_year}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def add_features(df: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Enhanced feature engineering using master dataset features

    Key improvements over baseline:
    1. GDD and chilling hours (phenology-specific)
    2. Climate indices (macro-climate drivers)
    3. Piecewise year splines (non-linear trends)
    4. Interaction terms (complex relationships)
    5. Hopkins Index and photoperiod (biological factors)
    """
    out = df.copy()
    ref = out if reference_df is None else reference_df.copy()

    # Site observation count (existing feature)
    if 'site_obs' not in out.columns:
        site_obs = ref.groupby('site_id').size().rename('site_obs').reset_index()
        out = out.merge(site_obs, on='site_id', how='left')
        out['site_obs'] = out['site_obs'].fillna(1)

    # Fill missing values with location or global medians
    # This handles sites with sparse weather data

    # Core meteorological features
    meteo_features = [
        'T_mean_winter', 'T_mean_spring', 'T_mean_jan_feb',
        'GDD_jan_feb', 'GDD_winter', 'GDD_nov_dec',
        'chill_hours_winter', 'chill_hours_nov_dec'
    ]

    # Climate indices
    climate_features = [
        'ONI_winter', 'NAO_winter', 'PDO_annual', 'AO_winter'
    ]

    # Phenology features (usually complete)
    pheno_features = [
        'Hopkins_Index', 'photoperiod_mar20', 'alt_log1p'
    ]

    # Interaction features
    interaction_features = [
        'lat_x_GDD', 'lat_x_chill', 'year_c_x_GDD'
    ]

    # Fill missing values
    all_features = meteo_features + climate_features

    for feature in all_features:
        if feature in out.columns and feature in ref.columns:
            # Location-level median
            loc_median = ref.groupby('location')[feature].median()
            out = out.merge(
                loc_median.rename(f'{feature}_loc').reset_index(),
                on='location',
                how='left'
            )

            # Global median
            global_median = ref[feature].median()

            # Fill: original -> location median -> global median -> 0
            out[feature] = (
                out[feature]
                .fillna(out[f'{feature}_loc'])
                .fillna(global_median)
                .fillna(0)
            )

            # Drop temporary column
            out = out.drop(columns=[f'{feature}_loc'], errors='ignore')

    # Ensure year-based features exist
    if 'year_c' not in out.columns:
        out['year_c'] = out['year'] - 1950

    if 'year_c2' not in out.columns:
        out['year_c2'] = out['year_c'] ** 2

    # Ensure piecewise splines exist
    if 'year_seg_1950' not in out.columns:
        out['year_seg_1950'] = np.maximum(0, out['year'] - 1950)
    if 'year_seg_1980' not in out.columns:
        out['year_seg_1980'] = np.maximum(0, out['year'] - 1980)
    if 'year_seg_2000' not in out.columns:
        out['year_seg_2000'] = np.maximum(0, out['year'] - 2000)

    # Ensure geographic features exist
    if 'lat_abs' not in out.columns:
        out['lat_abs'] = out['lat'].abs()

    if 'alt_log1p' not in out.columns:
        out['alt_log1p'] = np.log1p(np.clip(out['alt'], a_min=0, a_max=None))

    # Ensure Hopkins Index exists
    if 'Hopkins_Index' not in out.columns:
        out['Hopkins_Index'] = (
            4 * out['lat'] -
            1.25 * out['long'] +
            4 * (out['alt'] / 122)
        )

    # Ensure photoperiod exists
    if 'photoperiod_mar20' not in out.columns:
        out['photoperiod_mar20'] = 12.0  # Default to equinox

    # Ensure interaction terms exist
    if 'lat_x_GDD' not in out.columns and 'GDD_jan_feb' in out.columns:
        out['lat_x_GDD'] = out['lat'] * out['GDD_jan_feb']

    if 'lat_x_chill' not in out.columns and 'chill_hours_winter' in out.columns:
        out['lat_x_chill'] = out['lat'] * out['chill_hours_winter']

    if 'year_c_x_GDD' not in out.columns and 'GDD_jan_feb' in out.columns:
        out['year_c_x_GDD'] = out['year_c'] * out['GDD_jan_feb']

    return out

# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_enhanced_model() -> Pipeline:
    """
    Enhanced global model with meteorological and climate features

    Feature groups:
    1. Temporal: year, year splines
    2. Geographic: lat, long, alt
    3. Meteorological: temperature, GDD, chilling hours
    4. Climate indices: ONI, NAO, PDO, AO
    5. Phenology: Hopkins Index, photoperiod
    6. Interactions: lat×GDD, lat×chill, year×GDD
    7. Site characteristics: site_obs, source
    """

    # Core features
    num_cols = [
        # Temporal (non-linear trends via splines)
        'year', 'year_c', 'year_seg_1950', 'year_seg_1980', 'year_seg_2000',

        # Geographic
        'lat', 'long', 'alt_log1p', 'lat_abs',

        # Site characteristics
        'site_obs',

        # Meteorological (winter and spring temperatures)
        'T_mean_winter', 'T_mean_spring', 'T_mean_jan_feb',

        # Growing Degree Days (thermal accumulation)
        'GDD_jan_feb', 'GDD_winter', 'GDD_nov_dec',

        # Chilling hours (dormancy requirement)
        'chill_hours_winter', 'chill_hours_nov_dec',

        # Climate indices (macro-climate drivers)
        'ONI_winter',  # ENSO
        'NAO_winter',  # North Atlantic Oscillation
        'PDO_annual',  # Pacific Decadal Oscillation
        'AO_winter',   # Arctic Oscillation

        # Phenology/Biology
        'Hopkins_Index',
        'photoperiod_mar20',

        # Interaction terms
        'lat_x_GDD',
        'lat_x_chill',
        'year_c_x_GDD',
    ]

    cat_cols = ['source']

    # Preprocessing pipeline
    pre = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ],
        remainder='drop'
    )

    # Gradient Boosting Regressor (Huber loss for robustness)
    model = GradientBoostingRegressor(
        loss='huber',
        n_estimators=800,        # Increased from 700
        learning_rate=0.015,     # Slightly lower for better generalization
        max_depth=4,             # Increased from 3 for more complex interactions
        min_samples_split=5,     # Prevent overfitting
        min_samples_leaf=2,
        subsample=0.8,           # Stochastic gradient boosting
        max_features='sqrt',     # Random feature selection
        random_state=5103
    )

    return Pipeline([('pre', pre), ('model', model)])

def predict_local_trend(train_comp: pd.DataFrame, new_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Local trend model (Model A) - unchanged from baseline

    Uses recency-weighted quadratic regression for each site
    """
    rows = []
    for loc in new_comp['location'].unique():
        tr = train_comp[train_comp['location'] == loc].sort_values('year').copy()
        nd = new_comp[new_comp['location'] == loc].copy()

        n = len(tr)
        if n >= 4:
            # Exponential weights: more recent years get higher weight
            w = np.exp(np.arange(-n + 1, 1) / 6.0)
            coef = np.polyfit(tr['year'].values, tr['bloom_doy'].values, deg=2, w=w)
            pred = np.polyval(coef, nd['year'].values)
        elif n >= 2:
            # Linear trend for limited data
            coef = np.polyfit(tr['year'].values, tr['bloom_doy'].values, deg=1)
            pred = np.polyval(coef, nd['year'].values)
        else:
            # Mean for very limited data
            pred = np.repeat(tr['bloom_doy'].mean(), len(nd))

        nd['pred_local'] = pred
        rows.append(nd[['location', 'year', 'pred_local']])

    return pd.concat(rows, ignore_index=True)

# ============================================================================
# ROLLING BACKTEST
# ============================================================================

print("\n" + "="*80)
print("RUNNING ROLLING BACKTEST")
print("="*80)

backtest_start = max(1900, int(competition['year'].min()) + 20)
backtest_years = list(range(backtest_start, int(competition['year'].max()) + 1))

print(f"\nBacktest configuration:")
print(f"  Start year: {backtest_start}")
print(f"  End year: {backtest_years[-1]}")
print(f"  Total windows: {len(backtest_years)}")

rolling_rows = []

for y in backtest_years:
    # Training data: all years before y
    train_comp = competition[competition['year'] < y].copy()
    test_comp = competition[competition['year'] == y].copy()

    if test_comp.empty or train_comp['location'].nunique() < len(COMPETITION_SITES):
        continue

    # Global model training data (includes auxiliary)
    train_all = all_data[all_data['year'] < y].copy()
    train_all = add_features(train_all)

    # Test data preparation - ensure site_id exists
    test_comp_copy = test_comp.copy()
    if 'site_id' not in test_comp_copy.columns:
        test_comp_copy['site_id'] = test_comp_copy['source'] + '::' + test_comp_copy['location']
    test_feat = add_features(test_comp_copy, reference_df=train_all)

    # Model A: Local trend
    local_pred = predict_local_trend(train_comp, test_feat)

    # Model B: Enhanced global model
    g_model = build_enhanced_model()
    g_model.fit(train_all, train_all['bloom_doy'])
    pred_g = g_model.predict(test_feat)

    # Combine predictions
    fold = test_feat[['location', 'year', 'bloom_doy']].merge(
        local_pred, on=['location', 'year'], how='left'
    )
    fold['pred_global'] = pred_g
    rolling_rows.append(fold)

rolling = pd.concat(rolling_rows, ignore_index=True)

# Calculate MAE for each model
mae_local = mean_absolute_error(rolling['bloom_doy'], rolling['pred_local'])
mae_global = mean_absolute_error(rolling['bloom_doy'], rolling['pred_global'])

print(f"\nBacktest results:")
print(f"  Model A (local trend) MAE: {mae_local:.2f} days")
print(f"  Model B (enhanced global) MAE: {mae_global:.2f} days")

# ============================================================================
# ENSEMBLE WEIGHTING
# ============================================================================

print("\n" + "="*80)
print("OPTIMIZING ENSEMBLE WEIGHTS")
print("="*80)

# Grid search for optimal blending weight
grid = np.arange(0.0, 1.0001, 0.02)

def mae_w(df, w):
    pred = w * df['pred_local'] + (1.0 - w) * df['pred_global']
    return mean_absolute_error(df['bloom_doy'], pred)

# Global optimal weight
w_local_global = min(grid, key=lambda w: mae_w(rolling, w))
w_global_global = 1.0 - w_local_global

print(f"\nGlobal weights:")
print(f"  Local: {w_local_global:.2f}")
print(f"  Global: {w_global_global:.2f}")

# Site-specific weights
site_w = []
for loc, df_loc in rolling.groupby('location'):
    w_star = min(grid, key=lambda w: mae_w(df_loc, w))
    site_w.append({
        'location': loc,
        'w_local': w_star,
        'w_global': 1.0 - w_star
    })

site_w_df = pd.DataFrame(site_w)
print(f"\nSite-specific weights:")
print(site_w_df.to_string(index=False))

# Apply ensemble
rolling = rolling.merge(site_w_df, on='location', how='left')
rolling['w_local'] = rolling['w_local'].fillna(w_local_global)
rolling['w_global'] = rolling['w_global'].fillna(w_global_global)

rolling['pred_ensemble'] = (
    rolling['w_local'] * rolling['pred_local'] +
    rolling['w_global'] * rolling['pred_global']
)

rolling['abs_err'] = (rolling['bloom_doy'] - rolling['pred_ensemble']).abs()

mae_ensemble = mean_absolute_error(rolling['bloom_doy'], rolling['pred_ensemble'])

print(f"\nFinal ensemble MAE: {mae_ensemble:.2f} days")
print(f"Improvement over local: {mae_local - mae_ensemble:.2f} days")
print(f"Improvement over global: {mae_global - mae_ensemble:.2f} days")

# ============================================================================
# PREDICTION INTERVALS
# ============================================================================

print("\n" + "="*80)
print("CALCULATING PREDICTION INTERVALS")
print("="*80)

# Site-level 90th percentile of residuals
site_q90 = rolling.groupby('location', as_index=False)['abs_err'].quantile(0.90)
site_q90 = site_q90.rename(columns={'abs_err': 'q90'})

global_q90 = rolling['abs_err'].quantile(0.90)

print(f"\nGlobal 90th percentile: {global_q90:.2f} days")
print(f"\nSite-specific 90th percentiles:")
print(site_q90.to_string(index=False))

# ============================================================================
# 2026 PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print(f"GENERATING {target_year} PREDICTIONS")
print("="*80)

# Prepare training data (all available data)
train_all = add_features(all_data.copy())
train_comp = competition.copy()

# Prepare prediction data for target year
newdata = (
    competition
    .sort_values('year')
    .groupby('location', as_index=False)
    .tail(1)[['location', 'lat', 'long', 'alt', 'source', 'site_id']]
    .copy()
)

newdata['year'] = target_year
newdata['bloom_doy'] = np.nan

# Add features
new_feat = add_features(newdata, reference_df=train_all)

# Model A: Local trend
local_pred = predict_local_trend(train_comp, new_feat)

# Model B: Global model (retrain on all data)
global_model = build_enhanced_model()
global_model.fit(train_all, train_all['bloom_doy'])
pred_global = global_model.predict(new_feat)

# Ensemble predictions
final_pred = new_feat[['location', 'year']].merge(local_pred, on=['location', 'year'], how='left')
final_pred['pred_global'] = pred_global

# Apply weights
final_pred = final_pred.merge(site_w_df, on='location', how='left')
final_pred['w_local'] = final_pred['w_local'].fillna(w_local_global)
final_pred['w_global'] = final_pred['w_global'].fillna(w_global_global)

final_pred['prediction_raw'] = (
    final_pred['w_local'] * final_pred['pred_local'] +
    final_pred['w_global'] * final_pred['pred_global']
)

# Add intervals
final_pred = final_pred.merge(site_q90, on='location', how='left')
final_pred['q90'] = final_pred['q90'].fillna(global_q90)

final_pred['prediction'] = np.clip(np.round(final_pred['prediction_raw']), 1, 366).astype(int)
final_pred['lower'] = np.clip(np.floor(final_pred['prediction_raw'] - final_pred['q90']), 1, 366).astype(int)
final_pred['upper'] = np.clip(np.ceil(final_pred['prediction_raw'] + final_pred['q90']), 1, 366).astype(int)

final_pred = final_pred[['location', 'year', 'prediction', 'lower', 'upper']].sort_values('location')

print(f"\n{target_year} Predictions:")
print(final_pred.to_string(index=False))

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Convert DOY to dates
def doy_to_date(year: int, doy: int) -> pd.Timestamp:
    return pd.to_datetime(f'{year}-{doy:03d}', format='%Y-%j', errors='coerce')

submission = final_pred.copy()
submission['predicted_date'] = [doy_to_date(y, d) for y, d in zip(submission['year'], submission['prediction'])]
submission['lower_date'] = [doy_to_date(y, d) for y, d in zip(submission['year'], submission['lower'])]
submission['upper_date'] = [doy_to_date(y, d) for y, d in zip(submission['year'], submission['upper'])]

print("\nPredictions with dates:")
print(submission.to_string(index=False))

# Save submission file
output_file = 'cherry-predictions-enhanced.csv'
submission[['location', 'prediction', 'lower', 'upper']].to_csv(output_file, index=False)

print(f"\n✅ Saved predictions to: {output_file}")

# Save detailed results
results_file = 'enhanced_backtest_results.csv'
rolling.to_csv(results_file, index=False)
print(f"✅ Saved backtest results to: {results_file}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nEnhanced Model Performance:")
print(f"  Backtest MAE: {mae_ensemble:.2f} days")
print(f"  90th percentile error: {global_q90:.2f} days")
print(f"\nPredictions for {target_year}:")
for _, row in submission.iterrows():
    print(f"  {row['location']:15s}: {row['predicted_date'].strftime('%Y-%m-%d')} (DOY {row['prediction']})")
print(f"\n{'='*80}")
