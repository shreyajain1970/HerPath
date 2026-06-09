import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point

# ── Load street features ───────────────────────────────────────
print("Loading street features...")
streets = pd.read_csv('features_with_labels.csv')
print(f"Road segments: {len(streets)}")

# ── Load ward features ─────────────────────────────────────────
print("Loading ward features...")
wards = pd.read_csv('ward_features.csv')
print(f"Wards: {len(wards)}")
print(f"Ward columns: {wards.columns.tolist()}")

# ── Download ward boundaries from OSM ─────────────────────────
print("Downloading ward boundaries...")
ward_boundaries = ox.features_from_place(
    "Bengaluru, India",
    tags={"admin_level": "8"}
)
print(f"Ward boundaries found: {len(ward_boundaries)}")

# ── Convert street centroids to GeoDataFrame ──────────────────
print("Converting to GeoDataFrame...")
geometry = [
    Point(lng, lat) 
    for lat, lng in zip(
        streets['centroid_lat'], 
        streets['centroid_lng']
    )
]
streets_gdf = gpd.GeoDataFrame(
    streets, 
    geometry=geometry,
    crs="EPSG:4326"
)

# ── Spatial join — find which ward each road segment is in ─────
print("Performing spatial join...")
ward_boundaries = ward_boundaries.reset_index()

# Keep only relevant columns from ward boundaries
ward_boundaries_clean = ward_boundaries[
    ['name', 'geometry']
].copy()
ward_boundaries_clean.columns = ['ward_name_osm', 'geometry']

joined = gpd.sjoin(
    streets_gdf,
    ward_boundaries_clean,
    how='left',
    predicate='within'
)

print(f"Matched: {joined['ward_name_osm'].notna().sum()}")
print(f"Unmatched: {joined['ward_name_osm'].isna().sum()}")

# ── Merge crowd density from BBMP ward data ───────────────────
# Normalise ward names for matching
wards['ward_name_clean'] = wards['ward-name'].str.strip().str.lower()
joined['ward_name_clean'] = joined['ward_name_osm'].str.strip().str.lower()

merged = joined.merge(
    wards[['ward_name_clean', 'population_density']],
    on='ward_name_clean',
    how='left'
)

# Fill unmatched with median
median_density = merged['population_density'].median()
merged['crowd_density'] = merged['population_density'].fillna(
    median_density
)

print(f"\nCrowd density added:")
print(merged['crowd_density'].describe())

# ── Save final complete feature matrix ────────────────────────
final_cols = [
    'centroid_lat',
    'centroid_lng',
    'streetlight_score',
    'cctv_score',
    'police_proximity',
    'bus_stop_score',
    'crowd_density',
    'safety_label'
]

final_df = merged[final_cols].copy()
final_df.to_csv('features_complete.csv', index=False)
print(f"\nFinal dataset shape: {final_df.shape}")
print("Saved to features_complete.csv")
