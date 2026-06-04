# Pandora Time-Series Photometry Parquet Format Documentation

This document describes the schema and format of the time-series photometry product exported in Apache Parquet format by the Pandora data reduction pipeline.

## File Locations
- **Individual Target Consolidated files**: Saved in the output directory for each target, combining all exposures chronologically and detrending them together: `output_aux_pro/<target_name>_combined_photometry.parquet`.
- **Daily files**: Saved in the output directory for each day (grouped by target-level consolidated datasets), named `<YYYY_MM_DD>_photometry.parquet` (e.g. `output_aux_pro/2026_05_30_photometry.parquet`).
- **Consolidated file**: Created by merging daily files together using the helper script `merge_daily_parquet.py`, named `pandora_all_photometry.parquet` inside the output folder (e.g. `output_aux_pro/`).

---

## Schema & Column Definitions

Each Parquet file contains a tabular dataset with the following columns:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `target_id` | `string` | The identifier of the observed target star (e.g., `"WASP-167b"`, `"TOI-3235b"`, or Gaia ID). |
| `filename` | `string` | The base name of the FITS file from which this integration was extracted (e.g., `"2026-05-30__01-22-18_InfImg_WASP-167b_..."`). |
| `time_jd` | `float64` | The mid-integration Julian Date (JD) calculated dynamically from header time tags. |
| `flux_norm` | `float64` | The raw normalized white-light flux integrated over the extraction aperture (prior to detrending). |
| `flux_detrended`| `float64` | The systematic-corrected, normalized white-light flux. |
| `dx` | `float64` | Pointing drift offset along the spatial axis of the detector (in pixels). |
| `dy` | `float64` | Pointing drift offset along the dispersion axis of the detector (in pixels). |
| `pca_correction`| `float64` | The exact systematic correction factor applied to the flux (such that $\text{flux\_detrended} = \text{flux\_norm} / \text{pca\_correction}$). |
| `keep_mask` | `boolean` | Flag indicating whether the integration passed quality control checks (`True` if kept, `False` if flagged/rejected due to excursions or reacquisition). |
| `pc1` | `float64` | Score of the 1st principal component computed from standardized pointing drifts (`dx`, `dy`) and background level. |
| `pc2` | `float64` | Score of the 2nd principal component. |
| `pc3` | `float64` | Score of the 3rd principal component. |
| `flux_spec_{pixel}`| `float64` | The normalized, relative spectral flux at the specified `pixel` (dispersion pixel index, e.g., `flux_spec_73` to `flux_spec_284`). Each dispersion channel is stored as its own column. |

---

## Daily Parquet Merger Tool

You can merge daily Parquet files into a single consolidated catalog `pandora_all_photometry.parquet` using the helper script `merge_daily_parquet.py`:

```bash
# Merge daily files in the default output directory
./merge_daily_parquet.py

# Specify a custom directory containing the daily files
./merge_daily_parquet.py custom_output_dir/
```

---

## Querying and Analyzing Photometry

You can use the provided `query_photometry.py` script to filter and retrieve the photometry for a specific target.

### Usage
Run the script passing the target name as the first argument, and optionally the consolidated parquet file or the output directory:

```bash
# Query consolidated catalog (default)
./query_photometry.py WASP-167b

# Query using a specific daily or target Parquet file
./query_photometry.py TOI-3235b output_aux_pro/2026_05_30_photometry.parquet
```

---

## Example Python Usage

You can load and query the Parquet files directly in Python using `pandas`:

```python
import pandas as pd

# Load consolidated photometry
df = pd.read_parquet("output_aux_pro/pandora_all_photometry.parquet")

# Filter for WASP-167b and keep only good data points
wasp167 = df[(df["target_id"] == "WASP-167b") & (df["keep_mask"] == True)]

# Print basic statistics
print(f"Mean detrended flux: {wasp167['flux_detrended'].mean():.6f}")
print(f"Time span: {wasp167['time_jd'].max() - wasp167['time_jd'].min():.4f} days")

# Reconstruct the 2D Normalized Spectral Time Series matrix (shape: n_integrations x n_channels)
spec_cols = [col for col in wasp167.columns if col.startswith("flux_spec_")]
spec_cols = sorted(spec_cols, key=lambda c: int(c.split("_")[-1]))
spectral_matrix = wasp167[spec_cols].to_numpy()
print(f"Spectral matrix shape: {spectral_matrix.shape} (Integrations: {spectral_matrix.shape[0]}, Channels: {spectral_matrix.shape[1]})")
```
