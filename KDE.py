import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity

# ── Load Safecity incidents ────────────────────────────────────
safecity = pd.read_csv('bengaluru_safecity.csv')
safecity['latitude'] = pd.to_numeric(safecity['latitude'], errors='coerce')
safecity['longitude'] = pd.to_numeric(safecity['longitude'], errors='coerce')
safecity = safecity.dropna()

incident_coords = safecity[['latitude', 'longitude']].values

# ── Fit KDE on incident locations ─────────────────────────────
print("Fitting KDE on Safecity incidents...")
kde = KernelDensity(
    bandwidth=0.005,    # ~500 metres — tune this
    metric='haversine', # correct distance metric for lat/lng
    kernel='gaussian'
)
kde.fit(np.radians(incident_coords))

# ── Score every road segment ───────────────────────────────────
def get_safety_label(road_lat, road_lng, kde):
    """
    Returns safety label 0-1 for a road segment
    0 = very unsafe (high incident density)
    1 = very safe (low incident density)
    """
    point = np.radians([[road_lat, road_lng]])
    
    # log density — more negative = less dense = safer
    log_density = kde.score_samples(point)[0]
    
    # Convert to density
    density = np.exp(log_density)
    
    return density

# ── Apply to all road segments ─────────────────────────────────
print("Computing safety labels for all road segments...")
street_features = pd.read_csv('street_features.csv')

densities = []
for i, row in street_features.iterrows():
    if i % 10000 == 0:
        print(f"  Processing {i}/{len(street_features)}...")
    density = get_safety_label(
        row['centroid_lat'],
        row['centroid_lng'],
        kde
    )
    densities.append(density)

street_features['incident_density'] = densities

# ── Convert density to safety label ───────────────────────────
# Higher density = more incidents = less safe
# Invert and normalise to 0-1

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
street_features['safety_label'] = 1 - scaler.fit_transform(
    street_features[['incident_density']]
)

print(f"\nSafety label distribution:")
print(street_features['safety_label'].describe())

# Check how many segments are safe vs unsafe
print(f"\nSafe (>0.7): {(street_features['safety_label'] > 0.7).sum()}")
print(f"Moderate (0.5-0.7): {((street_features['safety_label'] >= 0.5) & (street_features['safety_label'] <= 0.7)).sum()}")
print(f"Risky (<0.5): {(street_features['safety_label'] < 0.5).sum()}")

# Save
street_features.to_csv('data/features_with_labels.csv', index=False)
print("\nSaved to data/features_with_labels.csv")
