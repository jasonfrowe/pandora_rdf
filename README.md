# Pandora RDF workbook

## Installation

1. Create a virtual environment.
2. Activate it.
3. Initialize the `bls_cuda` submodule and install it into the environment.
4. Install project dependencies, including `ipykernel` for notebook use.

```bash
python -m venv .pandora_rdf
source .pandora_rdf/bin/activate
python -m pip install --upgrade pip
git submodule update --init --recursive
python -m pip install -e ./bls_cuda
python -m pip install numba matplotlib astropy ipykernel
```

If you are cloning the repository for the first time, you can also fetch the submodule in one step:

```bash
git clone --recurse-submodules https://github.com/jasonfrowe/pandora_rdf.git
```

If you want this environment to appear as a notebook kernel:

```bash
python -m ipykernel install --user --name pandora_rdf --display-name "Python (pandora_rdf)"
```

## Data Reduction Pipeline

The `pandora_RDF_v2.py` script implements a comprehensive pipeline for reducing slitless spectroscopy data from the Pandora mission. The mission utilizes a spare JWST detector and employs read-up-ramp sampling to capture high-precision spectrophotometric data.

### Pipeline Overview

The pipeline transforms raw ramp data into detrended, science-ready light curves and spectra through the following stages:

1.  **Preprocessing & Ramp Fitting**:
    *   **Data Loading**: Reads raw science FITS files, including ramp cubes and auxiliary telemetry/time data.
    *   **Ramp Fitting**: Performs read-up-ramp fitting to produce slope (flux), intercept, and scatter maps.
2.  **Pixel-Level Cleaning**:
    *   **Bad Pixel Identification**: Detects hot pixels and bad pixel clumps based on ramp diagnostics.
    *   **Repair**: Applies neighbor-based interpolation to repair bad pixels and corrects detector edge effects.
    *   **Cosmic Ray Removal**: Identifies and corrects cosmic ray hits using temporal analysis across the data cube.
3.  **Spectral Extraction**:
    *   **Trace Estimation**: Automatically estimates the spectral trace and defines a variable-width aperture.
    *   **Spectrophotometry**: Extracts 1D spectra and computes "white-light" (integrated) flux.
    *   **Centroiding**: Tracks the spatial and dispersion centroids of the trace to monitor pointing drift.
4.  **Data Quality & Cleaning**:
    *   **Excursion Rejection**: Filters out integrations with anomalous flux or extreme pointing jumps.
    *   **Gap Masking**: Identifies and masks visibility gaps (e.g., Earth occultation, SAA) and initial "burn-in" frames.
    *   **Noise Budgeting**: Analyzes the noise components to assess the precision of the extracted data.
5.  **Systematic Detrending**:
    *   **PCA Decorrelation**: Uses Principal Component Analysis (PCA) on spacecraft telemetry (e.g., quaternions, position) and pointing metrics to model and remove systematic trends from the light curve.
    *   **Difference Imaging**: Implements a difference-imaging photometry approach to mitigate pointing-induced systematics by comparing frames to an aligned master reference.
6.  **Analysis & Validation**:
    *   **Transit Modeling**: Overlays physical transit models (via `pytfit5`) to validate the reduction and estimate residuals.
    *   **Multimetric Diagnostics**: Performs correlation analysis between the final flux and a wide array of telemetry and motion metrics to ensure no significant systematics remain.

This pipeline is designed to handle the specific characteristics of the JWST detector and the read-up-ramp sampling mode, ensuring that the final products are optimized for high-precision transit spectroscopy.