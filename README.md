# 🌲🔥 Forest Fire Mapping — India

A multi-step GPU-accelerated pipeline for mapping and analyzing forest fires across India using 
MODIS fire detections, NDVI vegetation indices, and Land Surface Temperature (LST) data.

---

## Overview

| Step | Focus | Output |
|------|-------|--------|
| **Step 1** | Forest fire point extraction | Fire points + annual/monthly CSVs |
| **Step 2** | NDVI stress index features | 9 vegetation-thermal metrics |
| **Step 3** | Land Surface Temperature analysis | Day/night LST + DTR trends + thermal footprint |
| **Step 4** | *Planned:* ML-based fire risk modeling | Risk predictions |

All steps:
- **Use the same study period:** Nov 1, 2000 – Dec 15, 2022
- **Use the same boundary:** `India_State_Boundary.shp` (dissolved national polygon)
- **Run on GPU:** CuPy with automatic CPU (NumPy) fallback
- **Operate at 1km resolution:** Aligned MODIS grids across FIRMS, LULC, NDVI, LST

---

## Step 1 — Forest Fire Points Extraction

**Notebook:** [`../FOREST_FIRE_POINTS_EXTRACTION(INDIA).ipynb`](../FOREST_FIRE_POINTS_EXTRACTION(INDIA).ipynb)

**Reference methodology:** Biswas et al. (2025) & Uthappa et al. (2025), forest
classes per Sannigrahi et al. (2018)

**Study period:** 1 Nov 2000 – 15 Dec 2022

### What it does

1. Loads the MODIS Collection 6.1 fire archive (FIRMS) for India.
2. Clips fire detections to India's *exact* polygon (not just a lon/lat bounding box).
3. Extracts the yearly ESA-CCI/C3S LULC ZIP archives and builds a binary forest mask per year.
4. For every fire point, looks up its LULC pixel and keeps only points on forest pixels.
5. Saves annual + monthly forest-fire CSVs, extraction summary, and plots.

**Compute bottleneck (GPU-accelerated):** ~124M-pixel LULC grid × up to ~300k fire points/year × 22 years.

### Data sources (Step 1)

| Data | Source | Included? |
|------|--------|-----------|
| MODIS fire archive (FIRMS, Collection 6.1) | [firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov/download/) | ❌ |
| LULC maps (ESA-CCI / C3S, 1992–2022) | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/satellite-land-cover) | ❌ |
| India state boundary | `India_State_Boundary.shp` | ✅ |

---

## Step 2 — NDVI Stress Index

**Notebook:** [`../NDVI_DATA_INDIA_/NDVI_ANALYSIS_WITH_FFP.ipynb`](../NDVI_DATA_INDIA_/NDVI_ANALYSIS_WITH_FFP.ipynb)

**Reference methodology:** Biswas et al. (2025), *Environ. Sci. Pollut. Res.*, 32:4856–4878

**Data:** MOD13A3.061, monthly 1km, NASA AppEEARS

### What it does

Derives 9 NDVI-based features from raw monthly NDVI time series:
1. QA-filtered NDVI mean
2. Climatological monthly mean (2001–2020 baseline)
3. Monthly anomaly
4. Trend (classical 2×12-MA, GPU-vectorized)
5. Residuals
6. Mann-Kendall τ (GPU-vectorized trend test)
7. CVSI (Cumulative Pre-Fire Vegetation Stress Index, optimal lag chosen via mutual information with Step 1 fire points)
8. LISA cluster map (Local Moran's I)
9. NDVI–fire breakpoint threshold (fitted on real Step 1 fire/no-fire labels)

**GPU acceleration:** Full-grid anomaly, decomposition, Mann-Kendall, and stress-index computation — 
infeasible with per-pixel serial loops, trivial on GPU.

---

## Step 3 — Land Surface Temperature (LST) Analysis

**Notebook:** [`LST_DAY_NIGHT.ipynb`](LST_DAY_NIGHT.ipynb)

**Reference methodology:** Biswas et al. (2025) & Uthappa et al. (2025)

**Data:** MOD11A2.061, 8-day 1km, NASA AppEEARS

### What it does

1. Loads MODIS MOD11A2.061 LST data (daytime + nighttime).
2. Applies QA filtering (keeps Good + Marginal reliability only).
3. Computes **Diurnal Temperature Range (DTR)** = LST Day − LST Night.
4. Builds climatology per pixel per month (2001–2020 baseline).
5. Computes anomalies (deviation from climatology).
6. **GPU-vectorized Mann-Kendall trend test** on day/night/DTR.
7. Rasterizes Step 1 forest-fire points onto the LST grid (thermal footprint).
8. Exports time series, spatial trends, and summary statistics.

**GPU acceleration:** Climatology, anomalies, Mann-Kendall τ, and rasterization all run 
on the full spatiotemporal grid simultaneously.

### Data sources (Step 3)

| Data | Source | Included? |
|------|--------|-----------|
| MODIS LST (MOD11A2.061, 8-day, 1km) | [NASA AppEEARS](https://appeears.earthdatacloud.nasa.gov/) | ❌ |
| Step 1 fire points (CSV) | Output of Step 1 | ✅ |
| India state boundary | `India_State_Boundary.shp` | ✅ |

---

## Getting Started with Step 3 (LST Analysis)

### 1. Download MODIS MOD11A2 data

1. Go to [https://appeears.earthdatacloud.nasa.gov/](https://appeears.earthdatacloud.nasa.gov/)
2. Register with NASA Earthdata (free account)
3. **Build a New Request:**
   - **Product:** `MOD11A2.061` (MODIS/Terra 8-Day LST)
   - **Temporal range:** `2000-11-01` to `2022-12-31`
   - **Spatial extent:** Upload `India_State_Boundary.shp` (or use bbox: lon [68, 97.5], lat [6.5, 37.5])
   - **Layers:** `LST_Day_1km`, `LST_Night_1km`, `QC_Day`, `QC_Night`
   - **Output format:** HDF
4. Download all HDF files when ready (~5–10 GB)

### 2. Organize data

Create folder in project root:
```
LST_DATA(MODIS MOD11A2)/
  ├── MOD11A2.A2000305.h25v08.061.*.hdf
  ├── MOD11A2.A2000313.h25v08.061.*.hdf
  ├── ... (all HDF files)
```

### 3. Run the notebook

Open [`LST_DAY_NIGHT.ipynb`](LST_DAY_NIGHT.ipynb), verify paths in Step 2 (Configuration), 
then run all cells top-to-bottom.

---

## Output Structure

```
Base folder: D:\FOREST FIRE MAPPING(INDIA)\

├── India_State_Boundary.shp/shx        ← Boundary shapefile (all steps)
├── FOREST_FIRE_POINTS_EXTRACTION(INDIA).ipynb    ← Step 1
├── Forest_Fire_Outputs/                ← Step 1 results
│   ├── all_forest_fire_points_2000-2022.csv
│   ├── annual_*.csv
│   ├── monthly_*.csv
│   └── ...
│
├── NDVI_DATA_INDIA_/                   ← Step 2 folder
│   ├── NDVI_ANALYSIS_WITH_FFP.ipynb
│   ├── NDVI_Fire_Susceptibility_Outputs/
│   │   ├── NDVI_*.csv
│   │   ├── CVSI_*.csv
│   │   └── ...
│   └── ...
│
└── LST_analysis/                        ← Step 3 (this step)
    ├── LST_DAY_NIGHT.ipynb
    ├── India_State_Boundary.shp/shx
    ├── LST_DATA(MODIS MOD11A2)/         ← ✨ Place HDF downloads here
    │   └── MOD11A2.A*.hdf
    └── LST_Outputs/                     ← Results
        ├── LST_temporal_statistics.csv
        ├── LST_trend_summary.csv
        └── LST_Summary_Analysis.png
```

---

## Integration: Multi-factor Fire Risk

**Step 1 output** (forest fire points) → **Step 2** (NDVI stress) & **Step 3** (LST thermal) → 
**Step 4** (ML risk model)

All three steps are pixel-aligned at 1km resolution on the same grid, same dates, same country boundary.
This enables direct correlation of:
- Forest fire occurrence (Step 1)
- Vegetation stress (Step 2)
- Thermal anomalies (Step 3)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No LST data found in study period" | Verify HDF files are in `LST_DATA(MODIS MOD11A2)/` with naming pattern `MOD11A2.A*.hdf` |
| GPU out of memory (OOM) | Reduce study period or download fewer tiles; CPU fallback still works |
| HDF read errors | Verify HDF files are not corrupted; try downloading again |
| Boundary mismatch with Steps 1–2 | Confirm you used same `India_State_Boundary.shp` (not Country boundary) |
| Fire CSV not found | Verify Step 1 has been run and `Forest_Fire_Outputs/all_forest_fire_points_2000-2022.csv` exists |

---

## References

- Biswas et al. (2025). *Environmental Science & Pollution Research*, 32:4856–4878.
- Uthappa et al. (2025). (Refer to Step 1 notebook for full citation.)
- Sannigrahi et al. (2018). Forest class scheme for ESA-CCI LULC.
- [NASA AppEEARS](https://appeears.earthdatacloud.nasa.gov/) — MODIS data download portal
- [CuPy documentation](https://cupy.dev/) — GPU-accelerated NumPy-like arrays

### How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace "FOREST_FIRE_POINTS_EXTRACTION(INDIA).ipynb"
# or open it in Jupyter/VS Code and run all cells top to bottom
```

GPU acceleration is optional — install the commented-out CUDA packages in
`requirements.txt` if you have an NVIDIA GPU with CUDA 12.x; otherwise the
pipeline runs on CPU automatically, just slower.

### Outputs (`Forest_Fire_Outputs/`)

```
Forest_Fire_Outputs/
├── all_fire_india_merged.csv          # all MODIS fire points clipped to India, 2000–2022
├── all_forest_fires_2000_2022.csv     # combined forest-only fire points
├── extraction_summary.csv             # one row per year — the headline results (tracked)
├── boundary/                          # dissolved India boundary (GeoPackage)
├── lulc_extracted/                    # extracted yearly LULC NetCDFs
├── forest_fire_points/                # one CSV per year
├── monthly_records/<year>/            # one CSV per year-month
└── plots/                             # 6 summary PNGs (tracked)
```

Only `extraction_summary.csv` and `plots/` are tracked in git — everything
else is large and reproducible by re-running the notebook (see `.gitignore`).

### Latest results (23 years, 2000–2022)

- **1,599,466** total MODIS fire points inside India after boundary clip + dedup
- **541,545** of those fall on forest LULC pixels
- Forest cover held steady at **~9.9–10.4%** of India's land area over the period
- Peak year: **2021** (111,467 total fire points, 38,116 forest fires)

| Year | Total fires | Forest fires | Forest fire % | Forest cover % |
|---:|---:|---:|---:|---:|
| 2000 | 1,419 | 225 | 15.86 | 9.86 |
| 2001 | 18,844 | 6,007 | 31.88 | 9.95 |
| 2002 | 26,939 | 4,290 | 15.92 | 9.95 |
| 2003 | 56,119 | 22,348 | 39.82 | 9.97 |
| 2004 | 64,757 | 28,668 | 44.27 | 10.01 |
| 2005 | 63,737 | 22,270 | 34.94 | 10.00 |
| 2006 | 66,195 | 27,241 | 41.15 | 10.00 |
| 2007 | 75,413 | 29,008 | 38.47 | 10.01 |
| 2008 | 71,025 | 23,356 | 32.88 | 10.01 |
| 2009 | 90,500 | 40,155 | 44.37 | 10.01 |
| 2010 | 76,894 | 31,108 | 40.46 | 10.01 |
| 2011 | 72,441 | 22,513 | 31.08 | 10.02 |
| 2012 | 93,536 | 35,214 | 37.65 | 9.91 |
| 2013 | 71,219 | 22,245 | 31.23 | 10.02 |
| 2014 | 76,638 | 23,844 | 31.11 | 10.02 |
| 2015 | 68,553 | 20,258 | 29.55 | 10.02 |
| 2016 | 89,354 | 28,903 | 32.35 | 10.13 |
| 2017 | 82,729 | 26,462 | 31.99 | 10.17 |
| 2018 | 91,342 | 29,028 | 31.78 | 10.19 |
| 2019 | 75,693 | 21,158 | 27.95 | 10.27 |
| 2020 | 76,149 | 16,421 | 21.56 | 10.30 |
| 2021 | 111,467 | 38,116 | 34.19 | 10.33 |
| 2022 | 78,503 | 22,707 | 28.93 | 10.43 |

All 27 study years (2000–2022) now use **real** ESA-CCI/C3S LULC data — no
nearest-year fallback is used anywhere in this range.

### Repo structure

```
.
├── FOREST_FIRE_POINTS_EXTRACTION(INDIA).ipynb  # Step 1 pipeline (this notebook)
├── build_notebook.py                            # script that generates the notebook programmatically
├── India_State_Boundary.shp / .shx               # India boundary used for clipping/plotting
├── Forest_Fire_Outputs/                          # pipeline outputs (partially tracked, see above)
├── requirements.txt
└── .gitignore
```

## Citation

- Biswas, S. et al. (2025). *[Forest fire detection methodology — see notebook
  header for full reference]*
- Uthappa, A. et al. (2025).
- Sannigrahi, S. et al. (2018). ESA-CCI/C3S forest land-cover class mapping.

## License

No license has been chosen yet for this repository's code. The MODIS fire
archive (NASA FIRMS) and ESA-CCI/C3S LULC data are subject to their own
respective data-use terms — see the source links above.
