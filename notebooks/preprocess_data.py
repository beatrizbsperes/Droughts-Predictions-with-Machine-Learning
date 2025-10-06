import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv('your_drought_data.csv')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Filter for FIPS 6107 only
df_filtered = df[df['fips'] == 6107].copy()

# Sort by date to ensure proper rolling calculations
df_filtered = df_filtered.sort_values('date').reset_index(drop=True)

# Identify rows with scores (weekly observations)
scored_dates = df_filtered[df_filtered['score'].notna()]['date'].values

# Create the new dataset - start with scored weeks only
result_rows = []

for target_date in scored_dates:
    target_date = pd.Timestamp(target_date)
    
    # Get the score for this week
    score_value = df_filtered[df_filtered['date'] == target_date]['score'].values[0]
    
    # Create masks for different rolling windows ending on target_date
    mask_7d = (df_filtered['date'] <= target_date) & (df_filtered['date'] > target_date - pd.Timedelta(days=7))
    mask_30d = (df_filtered['date'] <= target_date) & (df_filtered['date'] > target_date - pd.Timedelta(days=30))
    mask_90d = (df_filtered['date'] <= target_date) & (df_filtered['date'] > target_date - pd.Timedelta(days=90))
    mask_180d = (df_filtered['date'] <= target_date) & (df_filtered['date'] > target_date - pd.Timedelta(days=180))
    
    # Initialize row with basic info
    row = {
        'fips': 6107,
        'date': target_date,
        'score': score_value
    }
    
    # PRECIPITATION FEATURES - Sum over windows
    row['prec_sum_7d'] = df_filtered.loc[mask_7d, 'PRECTOT'].sum()
    row['prec_sum_30d'] = df_filtered.loc[mask_30d, 'PRECTOT'].sum()
    row['prec_sum_90d'] = df_filtered.loc[mask_90d, 'PRECTOT'].sum()
    row['prec_sum_180d'] = df_filtered.loc[mask_180d, 'PRECTOT'].sum()
    
    # TEMPERATURE FEATURES - Mean over windows
    for period, mask in [('7d', mask_7d), ('30d', mask_30d), ('90d', mask_90d), ('180d', mask_180d)]:
        row[f't2m_mean_{period}'] = df_filtered.loc[mask, 'T2M'].mean()
        row[f't2m_max_mean_{period}'] = df_filtered.loc[mask, 'T2M_MAX'].mean()
        row[f't2m_min_mean_{period}'] = df_filtered.loc[mask, 'T2M_MIN'].mean()
        row[f't2m_range_mean_{period}'] = df_filtered.loc[mask, 'T2M_RANGE'].mean()
        row[f'ts_mean_{period}'] = df_filtered.loc[mask, 'TS'].mean()
    
    # TEMPERATURE MAX for heatwave detection (7d and 30d only)
    row['t2m_max_7d'] = df_filtered.loc[mask_7d, 'T2M_MAX'].max()
    row['t2m_max_30d'] = df_filtered.loc[mask_30d, 'T2M_MAX'].max()
    
    # HUMIDITY PROXIES - Mean over 7/30/90d
    for period, mask in [('7d', mask_7d), ('30d', mask_30d), ('90d', mask_90d)]:
        row[f'qv2m_mean_{period}'] = df_filtered.loc[mask, 'QV2M'].mean()
        row[f't2mdew_mean_{period}'] = df_filtered.loc[mask, 'T2MDEW'].mean()
        row[f't2mwet_mean_{period}'] = df_filtered.loc[mask, 'T2MWET'].mean()
    
    # WIND FEATURES - Mean and Max over 7/30d (evaporative demand)
    for period, mask in [('7d', mask_7d), ('30d', mask_30d)]:
        # 10m wind
        row[f'ws10m_mean_{period}'] = df_filtered.loc[mask, 'WS10M'].mean()
        row[f'ws10m_max_{period}'] = df_filtered.loc[mask, 'WS10M_MAX'].max()
        row[f'ws10m_min_mean_{period}'] = df_filtered.loc[mask, 'WS10M_MIN'].mean()
        row[f'ws10m_range_mean_{period}'] = df_filtered.loc[mask, 'WS10M_RANGE'].mean()
        
        # 50m wind
        row[f'ws50m_mean_{period}'] = df_filtered.loc[mask, 'WS50M'].mean()
        row[f'ws50m_max_{period}'] = df_filtered.loc[mask, 'WS50M_MAX'].max()
        row[f'ws50m_min_mean_{period}'] = df_filtered.loc[mask, 'WS50M_MIN'].mean()
        row[f'ws50m_range_mean_{period}'] = df_filtered.loc[mask, 'WS50M_RANGE'].mean()
    
    # PRESSURE FEATURES - Mean over 7/30d
    row['ps_mean_7d'] = df_filtered.loc[mask_7d, 'PS'].mean()
    row['ps_mean_30d'] = df_filtered.loc[mask_30d, 'PS'].mean()
    
    # ADDITIONAL DROUGHT-RELEVANT FEATURES
    # Precipitation deficit (compare recent to longer-term average)
    row['prec_deficit_30v90d'] = row['prec_sum_30d'] - (row['prec_sum_90d'] / 3)
    row['prec_deficit_7v30d'] = row['prec_sum_7d'] - (row['prec_sum_30d'] / 4.3)
    
    # Temperature anomaly (recent vs longer-term)
    row['temp_anomaly_7v90d'] = row['t2m_mean_7d'] - row['t2m_mean_90d']
    row['temp_anomaly_30v180d'] = row['t2m_mean_30d'] - row['t2m_mean_180d']
    
    # Vapor pressure deficit proxy (temperature - dewpoint)
    row['vpd_proxy_7d'] = row['t2m_mean_7d'] - row['t2mdew_mean_7d']
    row['vpd_proxy_30d'] = row['t2m_mean_30d'] - row['t2mdew_mean_30d']
    
    result_rows.append(row)

# Create final dataframe
df_drought_features = pd.DataFrame(result_rows)

# Save to CSV
df_drought_features.to_csv('fips_6107_drought_features.csv', index=False)

print(f"Dataset created with {len(df_drought_features)} weekly observations")
print(f"Total features: {len(df_drought_features.columns)}")
print(f"\nFeature columns:")
print(df_drought_features.columns.tolist())
print(f"\nFirst few rows:")
print(df_drought_features.head())
print(f"\nDataset info:")
print(df_drought_features.info())
print(f"\nMissing values:")
print(df_drought_features.isnull().sum().sum())