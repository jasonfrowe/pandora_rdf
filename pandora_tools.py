import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from astropy.io import fits
import astropy.time as atime
import importlib

import pandas as pd

# Import Pandora Library Functions
import pandora_funcs
pandora = importlib.reload(pandora_funcs)

# Check if pytfit5 is available for transit modeling
try:
    import pytfit5.transitmodel as tm
    HAS_PYTFIT5 = True
except ImportError:
    HAS_PYTFIT5 = False
    print("WARNING: pytfit5.transitmodel not found. Transit fitting will be skipped.")

# Typical NIRDA/H2RG values
READ_NOISE = 12.0  # e-
GAIN = 1.5         # e-/DN

def get_planet_params(target_name):
    """
    Look up transit and ephemeris parameters for known targets.
    Returns a dictionary of parameters or None if unknown/reference.
    """
    if not target_name:
        return None
    
    # Standardize target name: lowercase and strip non-alphanumeric chars
    name_clean = "".join(c for c in target_name.lower() if c.isalnum())
    
    if "wasp178" in name_clean:
        return {
            "name": "WASP-178b",
            "ephem": {
                "period_days": 3.3448285,
                "t0_jd": 2456927.06839 - 0.0055,  # Convert HJD to JD approximately
                "t14_hours": 3.470,
            },
            "transit": {
                "a_over_Rstar": 7.17,
                "b": 0.54,
                "rp_over_Rstar": 0.11066,
                "u1": 0.10,
                "u2": 0.20,
            }
        }
    elif "toi3235" in name_clean:
        # TOI-3235b parameters from Hobson et al. 2023
        return {
            "name": "TOI-3235b",
            "ephem": {
                "period_days": 2.59261842,
                "t0_jd": 2459690.00173,  # BJD
                "t14_hours": 1.48,
            },
            "transit": {
                "a_over_Rstar": 15.75,
                "b": 0.511,
                "rp_over_Rstar": 0.2828,  # sqrt(0.079912)
                "u1": 0.15,
                "u2": 0.25,
            }
        }
    elif "wasp167" in name_clean or "kelt13" in name_clean:
        # WASP-167b parameters from Temple et al. 2017
        return {
            "name": "WASP-167b",
            "ephem": {
                "period_days": 2.0219596,
                "t0_jd": 2458582.07235,  # BJD
                "t14_hours": 2.724,
            },
            "transit": {
                "a_over_Rstar": 4.18,
                "b": 0.77,
                "rp_over_Rstar": 0.1001,
                "u1": 0.10,
                "u2": 0.20,
            }
        }
    
    return None

def load_pandora_data(fits_path):
    """
    Loads raw science data and timing information from a Pandora FITS file.
    Supports both the old 4D RAW SCIENCE format and the new 3D SCIENCE format.
    """
    with fits.open(fits_path, memmap=False) as hdul:
        primary_header = hdul[0].header
        target = primary_header.get("TARG_ID", "Unknown")
        
        # Check if the science array is in the old format or new format
        if "RAW SCIENCE" in hdul:
            print("Detected old format with RAW SCIENCE extension.")
            ramp_cube = np.asarray(hdul["RAW SCIENCE"].data)
            science_header = hdul["RAW SCIENCE"].header.copy()
            nint, ngroup = ramp_cube.shape[0], ramp_cube.shape[1]
            flat_cube = pandora.flatten_ramp_cube(ramp_cube)
            
            # Read auxiliary data and time extensions
            _, _, exposure_time_s = pandora.get_rdf_auxiliary_data(fits_path)
            time_jd_cube = pandora.read_rdf_time_extension(fits_path)
            integration_jd = np.asarray(time_jd_cube[:, -1], dtype=float)
            
        elif "SCIENCE" in hdul:
            print("Detected new 3D SCIENCE format.")
            flat_cube = np.asarray(hdul["SCIENCE"].data, dtype=float)
            science_header = hdul["SCIENCE"].header.copy()
            
            # Read size parameter from primary header
            nint = primary_header.get("INTEGRTS", 1)
            ngroup = primary_header.get("GRPS", 1)
            
            # If the exposure was interrupted, we may have fewer frames than expected.
            # Slice to only include complete integrations.
            nint_actual = flat_cube.shape[0] // ngroup
            if nint_actual < nint:
                print(f"WARNING: File contains {flat_cube.shape[0]} frames, which is fewer than expected {nint * ngroup} (INTEGRTS={nint}, GRPS={ngroup}).")
                print(f"Slicing to {nint_actual} complete integrations ({nint_actual * ngroup} frames).")
                flat_cube = flat_cube[:nint_actual * ngroup]
                nint = nint_actual
                
            # Reshape flat_cube (n_frames, ny, nx) to ramp_cube (nint, ngroup, ny, nx)
            ny, nx = flat_cube.shape[1], flat_cube.shape[2]
            ramp_cube = flat_cube.reshape(nint, ngroup, ny, nx)
            
            # Derive exposure times per group dynamically
            frmtime = primary_header.get("FRMTIME", 369.0)     # ms
            reads = primary_header.get("READS", 4)
            drops1 = primary_header.get("DROPS1", 1)
            drops2 = primary_header.get("DROPS2", 16)
            
            # Group time in seconds from integration start
            group_times = (drops1 + (reads - 1) / 2.0 + np.arange(ngroup) * (reads + drops2)) * frmtime * 1e-3
            exposure_time_s = np.tile(group_times, nint)
            
            # Derive integration JDs dynamically
            corstime = primary_header.get("CORSTIME", 0)
            finetime = primary_header.get("FINETIME", 0)
            frmstot = primary_header.get("FRMSTOT", nint * 106 + 50)
            resets1 = primary_header.get("RESETS1", 50)
            
            frames_per_integration = int((frmstot - resets1 + 1) // nint) if nint > 0 else 106
            
            # Start of observation in JD (J2000 epoch is 2451544.5)
            jd_sync = 2451544.5 + (corstime + finetime * 1e-9) / 86400.0
            
            # End of integration JDs (time at the last group)
            integration_jd = jd_sync + (resets1 - 1 + np.arange(nint) * frames_per_integration + drops1 + (reads - 1) / 2.0 + (ngroup - 1) * (reads + drops2)) * frmtime * 1e-3 / 86400.0
            
        else:
            raise KeyError("Neither RAW SCIENCE nor SCIENCE extension found in FITS file.")
            
    return ramp_cube, flat_cube, science_header, exposure_time_s, integration_jd, target

def run_pipeline(fits_path, output_dir="output", return_raw_extracted=False):
    """
    Executes the entire data reduction pipeline on a single Pandora FITS file.
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(fits_path))[0]
    
    print("\n" + "="*80)
    print(f"PROCESSING FILE: {fits_path}")
    print("="*80)
    
    # 1. Load Data
    ramp_cube, flat_cube, science_header, exposure_time_s, integration_jd, target = load_pandora_data(fits_path)
    nint, ngroup = ramp_cube.shape[0], ramp_cube.shape[1]
    
    print(f"Target: {target}")
    print(f"Datacube shape: {ramp_cube.shape} (nint={nint}, ngroup={ngroup})")
    print(f"JD range: ({np.nanmin(integration_jd):.6f}, {np.nanmax(integration_jd):.6f})")
    
    # Lookup planet parameters
    planet_params = get_planet_params(target)
    if planet_params:
        print(f"Found planet parameters for {planet_params['name']}:")
        print(f"  Period: {planet_params['ephem']['period_days']:.5f} days")
        print(f"  T0: {planet_params['ephem']['t0_jd']:.5f} JD")
        print(f"  Transit duration: {planet_params['ephem']['t14_hours']:.3f} hours")
    else:
        print("No planet parameters found for target. Fitting standard flat baseline.")
        
    # 2. Compute OWLS slope fit
    print("\nFitting ramps using OWLS...")
    slope_cube, intercept_cube, scatter_cube = pandora.get_slope_cube_owls(
        ramp_cube, times=exposure_time_s[0:ngroup], read_noise=READ_NOISE, gain=GAIN,
        threshold=4.0, return_diagnostics=True
    )
    
    # 3. Detect and repair bad/hot pixels
    print("Detecting and repairing bad pixels...")
    hot_pixel_mask, hot_info = pandora.detect_hot_pixels_from_ramp_fit(
        intercept_cube=intercept_cube,
        scatter_cube=scatter_cube,
        intercept_sigma=8.0,
        min_intercept_offset_dn=150.0,
        scatter_sigma=8.0,
    )
    
    badpix_params = {
        "min_intercept_offset_dn": 250.0,
        "intercept_sigma": 10.0,
        "slope_percentile": 2.0,
        "scatter_sigma": 10.0,
        "core_dilate_iterations": 0,
        "wing_iterations": 1,
        "repair_max_radius": 3,
        "repair_min_neighbors": 4,
        "repair_method": "local_plane",
        "repair_sigma_clip": 3.5,
        "edge_n_spatial_rows": 3,
        "edge_spatial_low": 0,
        "edge_spatial_high": slope_cube.shape[2] - 1,
        "edge_dispersion_low": 0,
        "edge_dispersion_high": slope_cube.shape[1] - 1,
    }
    
    bad_clump_mask, bad_clump_info = pandora.detect_bad_pixel_clumps_from_ramp_fit(
        intercept_cube=intercept_cube,
        slope_cube=slope_cube,
        scatter_cube=scatter_cube,
        min_intercept_offset_dn=badpix_params["min_intercept_offset_dn"],
        intercept_sigma=badpix_params["intercept_sigma"],
        slope_percentile=badpix_params["slope_percentile"],
        scatter_sigma=badpix_params["scatter_sigma"],
        dilate_iterations=badpix_params["core_dilate_iterations"],
    )
    
    # Filter out stellar trace from bad_clump_mask using high-slope cutoff
    pixel_slope = np.nanmedian(slope_cube, axis=0)
    median_slope = np.nanmedian(pixel_slope)
    mad_slope = np.nanmedian(np.abs(pixel_slope - median_slope))
    high_slope_cutoff = median_slope + 5.0 * (1.4826 * mad_slope)
    
    bad_clump_mask &= (~np.isfinite(pixel_slope) | (pixel_slope < high_slope_cutoff))
    bad_clump_info["n_clump_pixels"] = int(np.sum(bad_clump_mask))
    bad_clump_info["clump_fraction"] = float(np.mean(bad_clump_mask))
    
    repair_mask = pandora.dilate_binary_mask(
        bad_clump_mask,
        iterations=badpix_params["wing_iterations"],
    )
    wing_only_mask = repair_mask & (~bad_clump_mask)
    
    corrected_pixel_mask = repair_mask.copy()
    pixel_status_map = np.zeros(repair_mask.shape, dtype=np.uint8)
    hot_only_mask = hot_pixel_mask & (~corrected_pixel_mask)
    pixel_status_map[hot_only_mask] = 1
    pixel_status_map[bad_clump_mask] = 2
    pixel_status_map[wing_only_mask] = 3
    
    # Save Pixel Status Map
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = ListedColormap(["black", "gold", "red", "cyan"])
    im = ax.imshow(np.rot90(pixel_status_map), origin="lower", cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_title(f"Pixel Status Map ({target})")
    ax.set_xlabel("Dispersion pixel")
    ax.set_ylabel("Spatial pixel")
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels(["good", "hot-only", "core", "wing"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{basename}_pixel_status_map.png"), dpi=150)
    plt.close()
    
    # Repair slopes
    slope_cube_repaired, repair_info = pandora.correct_bad_pixels_with_neighbors(
        slope_cube,
        bad_mask=repair_mask,
        max_radius=badpix_params["repair_max_radius"],
        min_neighbors=badpix_params["repair_min_neighbors"],
        method=badpix_params["repair_method"],
        sigma_clip=badpix_params["repair_sigma_clip"],
    )
    
    slope_cube_repaired, edge_info = pandora.correct_detector_edge_pixels(
        slope_cube_repaired,
        n_spatial_edge_rows=badpix_params["edge_n_spatial_rows"],
        spatial_low=badpix_params["edge_spatial_low"],
        spatial_high=badpix_params["edge_spatial_high"],
        dispersion_low=badpix_params["edge_dispersion_low"],
        dispersion_high=badpix_params["edge_dispersion_high"],
    )
    
    # 4. Temporal Cosmic-Ray Correction
    print("Performing temporal cosmic-ray correction...")
    cr_params = {
        "window_frames": 9,
        "sigma": 5.0,
        "min_neighbors": 3,
        "positive_only": True,
    }
    slope_cube_cr_corrected, cr_mask, cr_info = pandora.correct_cosmic_rays_temporally(
        slope_cube_repaired,
        window_frames=cr_params["window_frames"],
        sigma=cr_params["sigma"],
        min_neighbors=cr_params["min_neighbors"],
        positive_only=cr_params["positive_only"],
    )
    
    # 5. Aperture / Trace Estimation
    print("Estimating trace and aperture...")
    trace_params = {
        "dispersion_min": 73,
        "dispersion_max": 284,
        "expected_spatial_center": 40,
        "max_half_width": 30,
        "smooth_window": 5,
        "threshold_sigma": 2.5,
        "spatial_left_start": 37.0,
        "spatial_left_end": 33.0,
        "spatial_right_start": 43.0,
        "spatial_right_end": 41.0,
    }
    
    trace_est = pandora.estimate_trace_aperture(
        slope_cube_cr_corrected,
        dispersion_min=trace_params["dispersion_min"],
        dispersion_max=trace_params["dispersion_max"],
        expected_spatial_center=trace_params["expected_spatial_center"],
        max_half_width=trace_params["max_half_width"],
        smooth_window=trace_params["smooth_window"],
        threshold_sigma=trace_params["threshold_sigma"],
    )
    
    aperture_model = pandora.build_linear_trace_aperture(
        n_dispersion=trace_est["dispersion_pixels"].size,
        dispersion_min=trace_params["dispersion_min"],
        dispersion_max=trace_params["dispersion_max"],
        spatial_left_start=trace_params["spatial_left_start"],
        spatial_left_end=trace_params["spatial_left_end"],
        spatial_right_start=trace_params["spatial_right_start"],
        spatial_right_end=trace_params["spatial_right_end"],
        n_spatial=slope_cube_cr_corrected.shape[2],
    )
    
    # Plot Aperture Overlay
    pandora.plot_spatial_profile(trace_est, trace_params)
    plt.savefig(os.path.join(output_dir, f"{basename}_spatial_profile.png"), dpi=150)
    plt.close()
    
    pandora.plot_aperture_overlay(slope_cube_cr_corrected, aperture_model, trace_est, image_index=min(44, nint - 1))
    plt.savefig(os.path.join(output_dir, f"{basename}_aperture_overlay.png"), dpi=150)
    plt.close()
    
    # Plot Sample Ramp
    sample_ramp_path = os.path.join(output_dir, f"{basename}_sample_ramp_fit.png")
    plot_sample_ramp(
        ramp_cube=ramp_cube,
        exposure_time_s=exposure_time_s,
        aperture_model=aperture_model,
        slope_cube=slope_cube,
        intercept_cube=intercept_cube,
        output_path=sample_ramp_path,
        integration_index=nint // 2,
        target_name=target
    )
    
    # 6. Spectrophotometric Extraction
    print("Extracting spectra...")
    extract_params = {
        "subtract_local_background": True,
        "background_inner_gap": 2,
        "background_width": 10,
        "use_wavelength_weights": True,
        "motion_aperture_guard_pixels": 2,
        "flux_window_low": 0.97,
        "flux_window_high": 1.03,
        "excursion_sigma": 6.0,
        "excursion_padding": 1,
        "motion_sigma": 6.0,
    }
    
    photometry_bad_mask = repair_mask
    extracted_spectra, extracted_dispersion, extraction_info = pandora.extract_trace_spectra_variable_aperture(
        slope_cube_cr_corrected,
        aperture_model=aperture_model,
        bad_pixel_mask=photometry_bad_mask,
        subtract_background=extract_params["subtract_local_background"],
        background_inner_gap=extract_params["background_inner_gap"],
        background_width=extract_params["background_width"],
        background_mask=photometry_bad_mask,
        return_diagnostics=True,
    )
    
    channel_good = pandora.build_channel_quality_mask(aperture_model, photometry_bad_mask)
    extracted_spectra_masked = extracted_spectra.copy()
    extracted_spectra_masked[:, ~channel_good] = np.nan
    
    # Predict event indices if planet is present
    transit_ingress_idx = 0
    transit_egress_idx = 0
    
    if planet_params:
        # Predict transit center and duration window
        jd_mid = np.nanmedian(integration_jd)
        epoch = int(np.rint((jd_mid - planet_params["ephem"]["t0_jd"]) / planet_params["ephem"]["period_days"]))
        event_center_jd = planet_params["ephem"]["t0_jd"] + epoch * planet_params["ephem"]["period_days"]
        event_ingress_jd = event_center_jd - 0.5 * planet_params["ephem"]["t14_hours"] / 24.0
        event_egress_jd = event_center_jd + 0.5 * planet_params["ephem"]["t14_hours"] / 24.0
        
        # Convert JDs to integration indices
        transit_ingress_idx = np.argmin(np.abs(integration_jd - event_ingress_jd))
        transit_egress_idx = np.argmin(np.abs(integration_jd - event_egress_jd))
        print(f"Predicted transit center: {event_center_jd:.6f} JD")
        print(f"Predicted transit indices: {transit_ingress_idx} to {transit_egress_idx}")
    else:
        # Reference targets: OOT mask is everything
        transit_ingress_idx = 0
        transit_egress_idx = -1
        
    wl = pandora.compute_white_light_products(
        extracted_spectra_masked, extraction_info, exposure_time_s, nint, ngroup,
        transit_ingress_index=transit_ingress_idx,
        transit_egress_index=transit_egress_idx,
        use_wavelength_weights=extract_params["use_wavelength_weights"],
    )
    
    integration_time_axis = wl["integration_time_axis"]
    time_axis_label      = wl["time_axis_label"]
    oot_mask             = wl["oot_mask"]
    white_light_norm     = wl["white_light_norm"]
    median_spectrum      = wl["median_spectrum"]
    normalized_spectra   = wl["normalized_spectra"]
    spectral_scatter_ppm = wl["spectral_scatter_ppm"]
    median_background_per_pixel = wl["median_background_per_pixel"]
    
    # Plot Normalized Spectral Time Series
    print("Plotting Normalized Spectral Time Series...")
    fig, ax = plt.subplots(figsize=(10, 6))
    vmin, vmax = np.nanpercentile(normalized_spectra, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.98, 1.02
        
    im = ax.imshow(
        normalized_spectra,
        origin="lower",
        aspect="auto",
        extent=[extracted_dispersion[0], extracted_dispersion[-1], 0, nint - 1],
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
    )
    ax.set_xlabel("Dispersion pixel", fontsize=10)
    ax.set_ylabel("Integration Index", fontsize=10)
    ax.set_title(f"Normalized Spectral Time Series: {target}\n(Outliers rejected: vmin={vmin:.3f}, vmax={vmax:.3f})", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Relative flux")
    plt.tight_layout()
    spec_time_series_path = os.path.join(output_dir, f"{basename}_spectral_time_series.png")
    plt.savefig(spec_time_series_path, dpi=150)
    plt.close()
    
    # 7. Spatial Motion Centroids and Excursion Rejection
    centroid_info = pandora.compute_aperture_motion_centroids(
        slope_cube_cr_corrected, aperture_model,
        background_per_pixel=extraction_info["background_per_pixel"],
        bad_pixel_mask=photometry_bad_mask,
        guard_pixels=int(extract_params["motion_aperture_guard_pixels"]),
    )
    dx = centroid_info["spatial_centroid"]    - np.nanmedian(centroid_info["spatial_centroid"][oot_mask])
    dy = centroid_info["dispersion_centroid"] - np.nanmedian(centroid_info["dispersion_centroid"][oot_mask])
    
    integration_keep_mask = pandora.reject_photometric_excursions(
        white_light_norm, dx, dy, oot_mask,
        flux_window_low=extract_params["flux_window_low"],
        flux_window_high=extract_params["flux_window_high"],
        excursion_sigma=extract_params["excursion_sigma"],
        excursion_padding=extract_params["excursion_padding"],
        motion_sigma=extract_params["motion_sigma"],
    )
    
    # Gap / reacquisition masking
    gap_params = {
        "gap_threshold_s": 100.0,
        "reacq_frames": 6,
        "initial_frames": 6,
        "manual_bad_indices": [],
    }
    
    integration_keep_mask[:gap_params["initial_frames"]] = False
    _dt_s = np.diff(integration_jd) * 86400.0
    _gap_starts = np.where(_dt_s > gap_params["gap_threshold_s"])[0]
    for _g in _gap_starts:
        _lo = int(_g) + 1
        _hi = min(nint, int(_g) + 1 + gap_params["reacq_frames"])
        integration_keep_mask[_lo:_hi] = False
        
    white_light_clean = white_light_norm.copy()
    white_light_clean[~integration_keep_mask] = np.nan
    
    if return_raw_extracted:
        raw_data = {
            "nint": nint,
            "target": target,
            "basename": basename,
            "fits_path": fits_path,
            "integration_jd": integration_jd,
            "white_light_norm": white_light_norm,
            "dx": dx,
            "dy": dy,
            "bg_per_int": np.nanmedian(extraction_info["background_per_pixel"], axis=1),
            "integration_keep_mask": integration_keep_mask,
            "oot_mask": oot_mask,
            "normalized_spectra": normalized_spectra,
            "extracted_dispersion": extracted_dispersion,
            "exposure_time_s": exposure_time_s,
            "ngroup": ngroup,
        }
        return raw_data
        
    # Save Spectrophotometry Diagnostics Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(integration_time_axis, white_light_norm, 'k.', alpha=0.3, label="raw")
    axes[0].plot(integration_time_axis[integration_keep_mask], white_light_clean[integration_keep_mask], 'g.', label="clean")
    axes[0].set_ylabel("WL Flux")
    axes[0].legend(loc="upper right")
    
    axes[1].plot(integration_time_axis, dx, 'r.-', label="dx")
    axes[1].plot(integration_time_axis, dy, 'b.-', label="dy")
    axes[1].set_ylabel("Drift (pix)")
    axes[1].legend(loc="upper right")
    
    axes[2].plot(integration_time_axis, np.nanmedian(extraction_info["background_per_pixel"], axis=1), 'm.-', label="background")
    axes[2].set_ylabel("Background")
    axes[2].set_xlabel(time_axis_label)
    axes[2].legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{basename}_extraction_diagnostics.png"), dpi=150)
    plt.close()
    
    # 8. Time and Motion Detrending (PCA without spacecraft telemetry)
    print("Detrending systematics (time + motion + background)...")
    _valid = integration_keep_mask.copy()
    _t_rel = integration_jd - np.nanmedian(integration_jd[_valid])
    _bg_per_int = np.nanmedian(extraction_info["background_per_pixel"], axis=1)
    
    _poly_cols = [_t_rel ** p for p in range(3)]  # 2nd order polynomial in time
    
    # Build design matrix with motion + background regressors
    _X_full = np.column_stack(_poly_cols + [dx, dy, _bg_per_int])
    
    _X_oot = _X_full[_valid & oot_mask]
    _y_oot = white_light_norm[_valid & oot_mask]
    
    if _X_oot.shape[0] < _X_oot.shape[1] + 1:
        print("WARNING: Too few OOT integrations for detrending; skipping.")
        white_light_detrended = white_light_clean.copy()
    else:
        _coef, _, _, _ = np.linalg.lstsq(_X_oot, _y_oot, rcond=None)
        _baseline_full = _X_full @ _coef
        
        _wl_valid = white_light_norm[_valid]
        _base_valid = _baseline_full[_valid]
        _oot_valid = oot_mask[_valid]
        
        _detrended = _wl_valid / _base_valid
        _detrended /= np.nanmedian(_detrended[_oot_valid])
        
        white_light_detrended = np.full(nint, np.nan)
        white_light_detrended[_valid] = _detrended
        
        detrended_oot_scatter_ppm = 1.0e6 * np.nanstd(_detrended[_oot_valid])
        raw_oot_scatter_ppm = 1.0e6 * np.nanstd(_wl_valid[_oot_valid])
        reduction_pct = 100.0 * (1.0 - (detrended_oot_scatter_ppm / raw_oot_scatter_ppm)) if raw_oot_scatter_ppm > 0 else np.nan
        print(f"OOT scatter before detrending: {raw_oot_scatter_ppm:.1f} ppm")
        print(f"OOT scatter  after detrending: {detrended_oot_scatter_ppm:.1f} ppm")
        print(f"Detrending reduced OOT scatter by {reduction_pct:.1f}%")
        
    # Save Detrended Lightcurve Plot
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(integration_jd, white_light_norm, color="0.6", s=6, label="Raw normalised")
    if _X_oot.shape[0] >= _X_oot.shape[1] + 1:
        ax.plot(integration_jd[_valid], _base_valid, color="tab:red", lw=1.2, alpha=0.7, label="Systematic model")
        ax.scatter(integration_jd, white_light_detrended - 0.05, color="tab:green", s=6, label="Detrended (−0.05 offset)")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    if _X_oot.shape[0] >= _X_oot.shape[1] + 1:
        ax.axhline(0.95, color="k", ls=":", lw=1)
    if planet_params:
        ax.axvline(event_center_jd, color="tab:blue", lw=1.2, ls="--", label="Predicted mid-transit")
        ax.axvspan(event_ingress_jd, event_egress_jd, color="tab:blue", alpha=0.14, label="Predicted T14")
    ax.set_xlabel("JD")
    ax.set_ylabel("Normalized white-light flux")
    ax.set_ylim(0.92, 1.05)
    ax.set_title(f"{target} — Detrended white-light curve")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{basename}_detrended_lightcurve.png"), dpi=150)
    plt.close()
    
    # 9. Transit Model Fit (if target has planet parameters and pytfit5 is installed)
    if planet_params and HAS_PYTFIT5:
        print("Fitting transit model...")
        _a_rs = planet_params["transit"]["a_over_Rstar"]
        _per_s = planet_params["ephem"]["period_days"] * 86400.0
        _rho_gcc = (_a_rs**3 * 3.0 * np.pi / (1000.0 * 6.674e-11 * _per_s**2))
        
        # Convert u1/u2 to Kipping (2013) q1/q2
        _u1, _u2 = planet_params["transit"]["u1"], planet_params["transit"]["u2"]
        _q1 = (_u1 + _u2) ** 2
        _q2 = _u1 / (2.0 * (_u1 + _u2))
        
        sol_transit = tm.transit_model_class()
        sol_transit.npl   = 1
        sol_transit.rho   = _rho_gcc
        sol_transit.nl3   = _q1
        sol_transit.nl4   = _q2
        sol_transit.t0    = [event_center_jd]
        sol_transit.per   = [planet_params["ephem"]["period_days"]]
        sol_transit.bb    = [planet_params["transit"]["b"]]
        sol_transit.rdr   = [planet_params["transit"]["rp_over_Rstar"]]
        sol_transit.ecw   = [0.0]
        sol_transit.esw   = [0.0]
        sol_transit.zpt   = 0.0
        sol_transit.dil   = 0.0
        
        _itime_days = np.asarray(exposure_time_s[::ngroup], dtype=float) / 86400.0
        _model_data = tm.transitModel(sol_transit, integration_jd, _itime_days, nintg=41)
        
        # Fine-sampled curve
        _jd_fine = np.linspace(integration_jd.min(), integration_jd.max(), 1000)
        _itime_fine = np.full(1000, float(np.nanmedian(_itime_days)))
        _model_fine = tm.transitModel(sol_transit, _jd_fine, _itime_fine, nintg=41)
        
        # Residuals
        _keep = integration_keep_mask
        _resid_detrended = white_light_detrended[_keep] - _model_data[_keep]
        _oot_keep = oot_mask[_keep]
        _resid_rms_ppm = 1.0e6 * np.nanstd(_resid_detrended[_oot_keep])
        
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax0 = axes[0]
        ax0.scatter(integration_jd, white_light_detrended, color="tab:orange", s=7, label="Detrended", zorder=2)
        ax0.plot(_jd_fine, _model_fine, color="tab:red", lw=1.5, label="Transit model", zorder=3)
        ax0.axhline(1.0, color="k", ls=":", lw=0.8)
        ax0.axvline(event_center_jd, color="tab:blue", lw=1.0, ls="--", alpha=0.6)
        ax0.axvspan(event_ingress_jd, event_egress_jd, color="tab:blue", alpha=0.10, label="Predicted T14")
        ax0.set_ylabel("Normalized flux")
        ax0.set_ylim(0.97, 1.03)
        ax0.set_title(f"{target} | a/R*={_a_rs:.2f}, b={planet_params['transit']['b']:.2f}, Rp/R*={planet_params['transit']['rp_over_Rstar']:.5f}")
        ax0.legend(loc="best", fontsize=8)
        
        ax1 = axes[1]
        ax1.scatter(integration_jd[_keep], _resid_detrended * 1.0e6, color="tab:orange", s=5, label="Residual")
        ax1.axhline(0.0, color="k", ls=":", lw=0.8)
        ax1.set_ylabel("Residual (ppm)")
        ax1.set_xlabel("JD")
        ax1.set_title(f"OOT residuals RMS = {_resid_rms_ppm:.1f} ppm")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{basename}_transit_model_fit.png"), dpi=150)
        plt.close()
        
    # 10. Save time-series photometry to Parquet
    import pandas as pd
    
    # Calculate exact applied systematic correction factor (pca_correction)
    pca_correction = np.ones(nint)
    valid_idx = np.isfinite(white_light_detrended) & (white_light_detrended > 0)
    pca_correction[valid_idx] = white_light_norm[valid_idx] / white_light_detrended[valid_idx]
    
    phot_data = {
        "target_id": [target] * nint,
        "filename": [os.path.basename(fits_path)] * nint,
        "time_jd": integration_jd.astype(np.float64),
        "flux_norm": white_light_norm.astype(np.float64),
        "flux_detrended": white_light_detrended.astype(np.float64),
        "dx": dx.astype(np.float64),
        "dy": dy.astype(np.float64),
        "pca_correction": pca_correction.astype(np.float64),
        "keep_mask": integration_keep_mask.astype(bool),
    }
    
    # Add per-channel normalized spectroscopy data
    for col_idx, disp_pix in enumerate(extracted_dispersion):
        phot_data[f"flux_spec_{disp_pix}"] = normalized_spectra[:, col_idx].astype(np.float64)
        
    phot_df = pd.DataFrame(phot_data)
    
    parquet_filename = f"{basename}_photometry.parquet"
    parquet_path = os.path.join(output_dir, parquet_filename)
    phot_df.to_parquet(parquet_path, index=False)
    print(f"Saved time-series photometry to {parquet_path}")
    
    print(f"Finished processing {target}. Output saved to {output_dir}/")
    print("="*80 + "\n")
    
    return phot_df


def run_target_pipeline(fits_paths, target_name, output_dir="output"):
    """
    Executes target-level consolidated processing.
    Collects raw photometry from each file, combines them, runs detrending/PCA,
    fits the transit model on the combined time-series, and returns the target dataframe.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*80)
    print(f"RUNNING CONSOLIDATED TARGET PIPELINE: {target_name}")
    print(f"Processing {len(fits_paths)} files...")
    print("="*80 + "\n")
    
    # 1. Run raw extraction for each file
    raw_data_list = []
    for f in fits_paths:
        try:
            raw = run_pipeline(f, output_dir=output_dir, return_raw_extracted=True)
            raw_data_list.append(raw)
        except Exception as e:
            print(f"FAILED raw extraction on {os.path.basename(f)}: {e}")
            import traceback
            traceback.print_exc()
            
    if not raw_data_list:
        raise ValueError(f"No successfully processed files for target {target_name}")
        
    # 2. Concatenate all outputs
    time_jd = np.concatenate([raw["integration_jd"] for raw in raw_data_list])
    white_light_norm = np.concatenate([raw["white_light_norm"] for raw in raw_data_list])
    dx = np.concatenate([raw["dx"] for raw in raw_data_list])
    dy = np.concatenate([raw["dy"] for raw in raw_data_list])
    bg_per_int = np.concatenate([raw["bg_per_int"] for raw in raw_data_list])
    integration_keep_mask = np.concatenate([raw["integration_keep_mask"] for raw in raw_data_list])
    oot_mask = np.concatenate([raw["oot_mask"] for raw in raw_data_list])
    normalized_spectra = np.concatenate([raw["normalized_spectra"] for raw in raw_data_list], axis=0)
    itime_days = np.concatenate([np.asarray(raw["exposure_time_s"][::raw["ngroup"]], dtype=float) / 86400.0 for raw in raw_data_list])
    
    # Create filenames array
    filenames = []
    for raw in raw_data_list:
        filenames.extend([raw["basename"] + ".fits"] * raw["nint"])
    filenames = np.array(filenames)
    
    # Check that extracted_dispersion is the same and store it
    extracted_dispersion = raw_data_list[0]["extracted_dispersion"]
    
    # 3. Sort chronologically by JD
    sort_idx = np.argsort(time_jd)
    time_jd = time_jd[sort_idx]
    white_light_norm = white_light_norm[sort_idx]
    dx = dx[sort_idx]
    dy = dy[sort_idx]
    bg_per_int = bg_per_int[sort_idx]
    integration_keep_mask = integration_keep_mask[sort_idx]
    oot_mask = oot_mask[sort_idx]
    normalized_spectra = normalized_spectra[sort_idx]
    itime_days = itime_days[sort_idx]
    filenames = filenames[sort_idx]
    
    nint = len(time_jd)
    print(f"Combined dataset shape: {nint} integrations across {len(fits_paths)} exposures.")
    
    # 4. Detrending on the combined dataset
    _valid = integration_keep_mask.copy()
    _t_rel = time_jd - np.nanmedian(time_jd[_valid])
    _poly_cols = [_t_rel ** p for p in range(3)]
    _X_full = np.column_stack(_poly_cols + [dx, dy, bg_per_int])
    
    _X_oot = _X_full[_valid & oot_mask]
    _y_oot = white_light_norm[_valid & oot_mask]
    
    pca_correction = np.ones(nint)
    white_light_detrended = white_light_norm.copy()
    
    if _X_oot.shape[0] < _X_oot.shape[1] + 1:
        print("WARNING: Too few OOT integrations for combined detrending; skipping.")
    else:
        _coef, _, _, _ = np.linalg.lstsq(_X_oot, _y_oot, rcond=None)
        _baseline_full = _X_full @ _coef
        
        _wl_valid = white_light_norm[_valid]
        _base_valid = _baseline_full[_valid]
        _oot_valid = oot_mask[_valid]
        
        _detrended = _wl_valid / _base_valid
        _detrended /= np.nanmedian(_detrended[_oot_valid])
        
        white_light_detrended = np.full(nint, np.nan)
        white_light_detrended[_valid] = _detrended
        
        valid_idx = np.isfinite(white_light_detrended) & (white_light_detrended > 0)
        pca_correction[valid_idx] = white_light_norm[valid_idx] / white_light_detrended[valid_idx]
        
        detrended_oot_scatter_ppm = 1.0e6 * np.nanstd(_detrended[_oot_valid])
        raw_oot_scatter_ppm = 1.0e6 * np.nanstd(_wl_valid[_oot_valid])
        reduction_pct = 100.0 * (1.0 - (detrended_oot_scatter_ppm / raw_oot_scatter_ppm)) if raw_oot_scatter_ppm > 0 else np.nan
        print(f"Combined OOT scatter before detrending: {raw_oot_scatter_ppm:.1f} ppm")
        print(f"Combined OOT scatter  after detrending: {detrended_oot_scatter_ppm:.1f} ppm")
        print(f"Detrending reduced combined OOT scatter by {reduction_pct:.1f}%")
        
    # 5. PCA on drift & background metrics to get top 3 PCs
    X_pca = np.column_stack([dx, dy, bg_per_int])
    mu_pca = np.nanmean(X_pca, axis=0)
    sig_pca = np.nanstd(X_pca, axis=0)
    sig_pca = np.where(sig_pca > 0, sig_pca, 1.0)
    Z_pca = (X_pca - mu_pca[None, :]) / sig_pca[None, :]
    Z_pca_clean = np.where(np.isfinite(Z_pca), Z_pca, 0.0)
    
    U_pca, S_pca, Vt_pca = np.linalg.svd(Z_pca_clean, full_matrices=False)
    pc_scores = Z_pca_clean @ Vt_pca.T
    pc1 = pc_scores[:, 0] if pc_scores.shape[1] > 0 else np.zeros(nint)
    pc2 = pc_scores[:, 1] if pc_scores.shape[1] > 1 else np.zeros(nint)
    pc3 = pc_scores[:, 2] if pc_scores.shape[1] > 2 else np.zeros(nint)
    
    # 6. Save combined plots
    planet_params = get_planet_params(target_name)
    if planet_params:
        jd_mid = np.nanmedian(time_jd)
        epoch = int(np.rint((jd_mid - planet_params["ephem"]["t0_jd"]) / planet_params["ephem"]["period_days"]))
        event_center_jd = planet_params["ephem"]["t0_jd"] + epoch * planet_params["ephem"]["period_days"]
        event_ingress_jd = event_center_jd - 0.5 * planet_params["ephem"]["t14_hours"] / 24.0
        event_egress_jd = event_center_jd + 0.5 * planet_params["ephem"]["t14_hours"] / 24.0
    else:
        event_center_jd = np.nanmedian(time_jd)
        event_ingress_jd = event_center_jd
        event_egress_jd = event_center_jd
        
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(time_jd, white_light_norm, color="0.6", s=6, label="Raw normalised")
    if _X_oot.shape[0] >= _X_oot.shape[1] + 1:
        ax.plot(time_jd[_valid], _baseline_full[_valid] * np.nanmedian(white_light_norm[_valid & oot_mask] / _baseline_full[_valid & oot_mask]), color="tab:red", lw=1.2, alpha=0.7, label="Systematic model")
        ax.scatter(time_jd, white_light_detrended - 0.05, color="tab:green", s=6, label="Detrended (−0.05 offset)")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    if _X_oot.shape[0] >= _X_oot.shape[1] + 1:
        ax.axhline(0.95, color="k", ls=":", lw=1)
    if planet_params:
        ax.axvline(event_center_jd, color="tab:blue", lw=1.2, ls="--", label="Predicted mid-transit")
        ax.axvspan(event_ingress_jd, event_egress_jd, color="tab:blue", alpha=0.14, label="Predicted T14")
    ax.set_xlabel("JD")
    ax.set_ylabel("Normalized white-light flux")
    ax.set_ylim(0.92, 1.05)
    ax.set_title(f"{target_name} — Combined Detrended white-light curve")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{target_name}_combined_detrended_lightcurve.png"), dpi=150)
    plt.close()
    
    # Save combined extraction diagnostics
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    white_light_clean = white_light_norm.copy()
    white_light_clean[~integration_keep_mask] = np.nan
    axes[0].plot(time_jd, white_light_norm, 'k.', alpha=0.3, label="raw")
    axes[0].plot(time_jd[integration_keep_mask], white_light_clean[integration_keep_mask], 'g.', label="clean")
    axes[0].set_ylabel("WL Flux")
    axes[0].legend(loc="upper right")
    
    axes[1].plot(time_jd, dx, 'r.-', label="dx")
    axes[1].plot(time_jd, dy, 'b.-', label="dy")
    axes[1].set_ylabel("Drift (pix)")
    axes[1].legend(loc="upper right")
    
    axes[2].plot(time_jd, bg_per_int, 'm.-', label="background")
    axes[2].set_ylabel("Background")
    axes[2].set_xlabel("JD")
    axes[2].legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{target_name}_combined_extraction_diagnostics.png"), dpi=150)
    plt.close()
    
    # Fit combined transit model
    if planet_params and HAS_PYTFIT5:
        print("Fitting combined transit model...")
        _a_rs = planet_params["transit"]["a_over_Rstar"]
        _per_s = planet_params["ephem"]["period_days"] * 86400.0
        _rho_gcc = (_a_rs**3 * 3.0 * np.pi / (1000.0 * 6.674e-11 * _per_s**2))
        
        # Convert u1/u2 to Kipping (2013) q1/q2
        _u1, _u2 = planet_params["transit"]["u1"], planet_params["transit"]["u2"]
        _q1 = (_u1 + _u2) ** 2
        _q2 = _u1 / (2.0 * (_u1 + _u2))
        
        sol_transit = tm.transit_model_class()
        sol_transit.npl   = 1
        sol_transit.rho   = _rho_gcc
        sol_transit.nl3   = _q1
        sol_transit.nl4   = _q2
        sol_transit.t0    = [event_center_jd]
        sol_transit.per   = [planet_params["ephem"]["period_days"]]
        sol_transit.bb    = [planet_params["transit"]["b"]]
        sol_transit.rdr   = [planet_params["transit"]["rp_over_Rstar"]]
        sol_transit.ecw   = [0.0]
        sol_transit.esw   = [0.0]
        sol_transit.zpt   = 0.0
        sol_transit.dil   = 0.0
        
        _model_data = tm.transitModel(sol_transit, time_jd, itime_days, nintg=41)
        _jd_fine = np.linspace(time_jd.min(), time_jd.max(), 1000)
        _itime_fine = np.full(1000, float(np.nanmedian(itime_days)))
        _model_fine = tm.transitModel(sol_transit, _jd_fine, _itime_fine, nintg=41)
        
        _keep = integration_keep_mask
        _resid_detrended = white_light_detrended[_keep] - _model_data[_keep]
        _oot_keep = oot_mask[_keep]
        _resid_rms_ppm = 1.0e6 * np.nanstd(_resid_detrended[_oot_keep])
        
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        axes[0].scatter(time_jd, white_light_detrended, color="tab:orange", s=7, label="Detrended", zorder=2)
        axes[0].plot(_jd_fine, _model_fine, color="tab:red", lw=1.5, label="Transit model", zorder=3)
        axes[0].axhline(1.0, color="k", ls=":", lw=0.8)
        axes[0].axvline(event_center_jd, color="tab:blue", lw=1.0, ls="--", alpha=0.6)
        axes[0].axvspan(event_ingress_jd, event_egress_jd, color="tab:blue", alpha=0.10, label="Predicted T14")
        axes[0].set_ylabel("Normalized flux")
        axes[0].set_ylim(0.97, 1.03)
        axes[0].set_title(f"{target_name} Combined | a/R*={_a_rs:.2f}, b={planet_params['transit']['b']:.2f}, Rp/R*={planet_params['transit']['rp_over_Rstar']:.5f}")
        axes[0].legend(loc="best", fontsize=8)
        
        axes[1].scatter(time_jd[_keep], _resid_detrended * 1.0e6, color="tab:orange", s=5, label="Residual")
        axes[1].axhline(0.0, color="k", ls=":", lw=0.8)
        axes[1].set_ylabel("Residual (ppm)")
        axes[1].set_xlabel("JD")
        axes[1].set_title(f"OOT residuals RMS = {_resid_rms_ppm:.1f} ppm")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{target_name}_combined_transit_model_fit.png"), dpi=150)
        plt.close()
        
    # 7. Construct target-level DataFrame
    phot_data = {
        "target_id": [target_name] * nint,
        "filename": filenames,
        "time_jd": time_jd.astype(np.float64),
        "flux_norm": white_light_norm.astype(np.float64),
        "flux_detrended": white_light_detrended.astype(np.float64),
        "dx": dx.astype(np.float64),
        "dy": dy.astype(np.float64),
        "pca_correction": pca_correction.astype(np.float64),
        "keep_mask": integration_keep_mask.astype(bool),
        "pc1": pc1.astype(np.float64),
        "pc2": pc2.astype(np.float64),
        "pc3": pc3.astype(np.float64),
    }
    
    # Add per-channel normalized spectroscopy data
    for col_idx, disp_pix in enumerate(extracted_dispersion):
        phot_data[f"flux_spec_{disp_pix}"] = normalized_spectra[:, col_idx].astype(np.float64)
        
    phot_df = pd.DataFrame(phot_data)
    
    # Save combined photometry to Parquet
    parquet_path = os.path.join(output_dir, f"{target_name}_combined_photometry.parquet")
    phot_df.to_parquet(parquet_path, index=False)
    print(f"Saved combined time-series photometry to {parquet_path}")
    print(f"Finished processing target {target_name}.\n" + "="*80 + "\n")
    
    return phot_df



def plot_sample_ramp(
    ramp_cube,
    exposure_time_s,
    aperture_model,
    slope_cube,
    intercept_cube,
    output_path,
    integration_index=None,
    target_name=""
):
    """
    Plots the raw group ramp and the fitted line for a pixel inside the aperture.
    Draws a horizontal dashed line at the saturation limit (65535 DN) to validate
    that the spectrum is not saturated.
    """
    nint, ngroup, ny, nx = ramp_cube.shape
    if integration_index is None:
        integration_index = nint // 2
    else:
        integration_index = min(max(0, int(integration_index)), nint - 1)
        
    disp_pixels = aperture_model["dispersion_pixels"]
    dy = len(disp_pixels) // 2
    y = disp_pixels[dy]
    
    aperture_mask = aperture_model["aperture_mask"]
    valid_x = np.where(aperture_mask[dy])[0]
    if len(valid_x) > 0:
        x = valid_x[len(valid_x) // 2]
    else:
        x = int(np.round(aperture_model["spatial_center"][dy]))
        
    x = min(max(0, x), nx - 1)
    y = min(max(0, y), ny - 1)
    
    raw_ramp = ramp_cube[integration_index, :, y, x]
    times = exposure_time_s[:ngroup]
    
    slope = slope_cube[integration_index, y, x]
    intercept = intercept_cube[integration_index, y, x]
    fit_ramp = intercept + slope * times
    
    print(f"Sample ramp chosen: pixel (dispersion y={y}, spatial x={x}) for integration {integration_index}")
    print(f"  Slope: {slope:.3f} DN/s, Intercept: {intercept:.3f} DN")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Premium styling
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.scatter(times, raw_ramp, color="#1f77b4", edgecolor="k", s=50, label="Raw group data", zorder=3)
    ax.plot(times, fit_ramp, color="#d62728", lw=2, label=f"OWLS Fit (Slope = {slope:.1f} DN/s)", zorder=4)
    
    # Saturation limit line
    ax.axhline(65535.0, color="black", linestyle="--", lw=1.5, label="Saturation Limit (65535 DN)", zorder=2)
    
    ax.set_title(f"Sample Pixel Ramp Fit: {target_name}\nPixel (disp={y}, spat={x}) | Integration {integration_index}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time from integration start (s)", fontsize=10)
    ax.set_ylabel("Signal (DN)", fontsize=10)
    
    # Dynamic y-axis limits to always show saturation line and data nicely
    max_val = max(65535.0, np.max(raw_ramp))
    ax.set_ylim(-2000, max_val * 1.1)
    
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

