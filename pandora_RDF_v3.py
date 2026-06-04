# %%
import numpy as np
import importlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import time

# Import Pandora Library Functions with autoreload
import pandora_funcs

pandora = importlib.reload(pandora_funcs)

try:
	from IPython import get_ipython
	ip = get_ipython()
	if ip is not None:
		ip.run_line_magic('load_ext', 'autoreload')
		ip.run_line_magic('autoreload', '2')
except Exception:
	pass
# %%
# Typical NIRCam/NIRISS values (Adjust read_noise and gain based on your specific detector)
READ_NOISE = 12.0 # e-
GAIN = 1.5 # e-/DN
#%%
# location data products
datadir = '/opt/data2/rowe/pandora/2026/RDF1/data/'
science_file = 'Pandora_RDF_WASP-178b_all.fits'
fits_path = datadir + science_file
ramp_cube, science_header = pandora.read_rdf_raw_science(fits_path)
cube = pandora.flatten_ramp_cube(ramp_cube)
nint, ngroup = ramp_cube.shape[0], ramp_cube.shape[1]

row_map, col_map, exposure_time_s = pandora.get_rdf_auxiliary_data(fits_path)

print(f"Loaded RAW SCIENCE ramp with shape: {ramp_cube.shape} (nint={nint}, ngroup={ngroup})")
print(f"Flattened science cube shape: {cube.shape}")
print(f"SCIENCE EXTNAME: {science_header.get('EXTNAME', 'N/A')}")

if exposure_time_s is not None:
	print(
		f"Loaded EXPOSURE_TIME array: {exposure_time_s.shape}, "
		f"min={np.nanmin(exposure_time_s):.6g}s, max={np.nanmax(exposure_time_s):.6g}s"
	)

time_jd_cube = pandora.read_rdf_time_extension(fits_path)
integration_jd = np.asarray(time_jd_cube[:, -1], dtype=float)
print(
	f"Loaded TIME extension: {time_jd_cube.shape}, "
	f"jd range=({np.nanmin(time_jd_cube):.6f}, {np.nanmax(time_jd_cube):.6f})"
)
# %%
pandora.display_science_image(cube, image_index=269, scale_style="zscale", iraf_contrast=0.99)
# %%
cube.shape
# %%
def plot_ramp_at_pixel(ramp_cube, int_idx, x, y, group_times=None,
                       read_noise=10.0, gain=1.0, threshold=4.0):
    """
    Plot time (group) vs flux for a single pixel in a given integration,
    overlaying the OWLS-fitted ramp slope and intercept.

    Parameters
    ----------
    ramp_cube : ndarray
        Shape (n_int, n_group, nx, ny).
    int_idx : int
        Integration index to plot.
    x, y : int
        Pixel coordinates (X, Y) on the detector.
    group_times : array-like, optional
        Physical time (e.g. seconds) for each group.  If None, uses group index.
    read_noise, gain, threshold : float, optional
        Parameters passed to get_slope_cube_owls.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    flux = ramp_cube[int_idx, :, x, y]

    if group_times is not None:
        t = np.asarray(group_times)
        xlabel = "Time (s)"
    else:
        t = np.arange(len(flux))
        xlabel = "Group"

    # Compute OWLS fit
    slope_cube, intercept_cube, _ = pandora_funcs.get_slope_cube_owls(
        ramp_cube, times=t, read_noise=read_noise, gain=gain,
        threshold=threshold, return_diagnostics=True
    )

    slope_val = slope_cube[int_idx, x, y]
    intercept_val = intercept_cube[int_idx, x, y]

    # Fit line and residuals
    fit_line = slope_val * t + intercept_val
    residuals = flux - fit_line

    # Error bars: variance accumulates from zero-point (bias)
    n_groups = len(flux)
    err = (read_noise / gain) * np.sqrt(np.arange(n_groups) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(5, 6), sharex=True, gridspec_kw={'hspace': 0})
    ax_main, ax_resid = axes

    # Main plot
    ax_main.errorbar(t, flux, yerr=err, fmt='o', capsize=3, label='Data: Pixel (X={}, Y={})'.format(x, y), color='blue')
    ax_main.plot(t, fit_line, color='red', linestyle='--', linewidth=2,
                 label=f'OWLS Fit\nSlope: {slope_val:.3f}')
    ax_main.set_ylabel("Flux (DN)")
    ax_main.legend()

    # Residuals plot
    ax_resid.errorbar(t, residuals, yerr=err, fmt='o', capsize=3, color='blue')
    ax_resid.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax_resid.set_xlabel(xlabel)
    ax_resid.set_ylabel("Residuals (DN)")

    plt.setp(ax_main.get_xticklabels(), visible=False)
    fig.tight_layout()
    return fig, axes

# %%
fig, ax = plot_ramp_at_pixel(ramp_cube, int_idx=44, x=200, y=40, group_times=exposure_time_s[0:6], read_noise=READ_NOISE, gain=GAIN, threshold=4.0)
plt.show()
# %%
# Compute OWLS fit
slope_cube, intercept_cube, scatter_cube = pandora_funcs.get_slope_cube_owls(
    ramp_cube, times=exposure_time_s[0:6], read_noise=READ_NOISE, gain=GAIN,
    threshold=4.0, return_diagnostics=True
)
# %%
slope_cube.shape
# %%
pandora.display_science_image(slope_cube, image_index=44, scale_style="zscale", iraf_contrast=0.99)
# %%
hot_pixel_mask, hot_info = pandora.detect_hot_pixels_from_ramp_fit(
	intercept_cube=intercept_cube,
	scatter_cube=scatter_cube,
	intercept_sigma=8.0,
	min_intercept_offset_dn=150.0,
	scatter_sigma=8.0,
)

print(
	f"Hot pixels from ramp zero-point map: {hot_info['n_hot']} "
	f"({100.0 * hot_info['hot_fraction']:.4f}%)"
)
# %%

# Tunable bad-pixel/clump-repair settings.
badpix_params = {
	"min_intercept_offset_dn": 250.0,
	"intercept_sigma": 10.0,
	"slope_percentile": 2.0,  # Increased from 0.5 - less aggressive after timing fix
	"scatter_sigma": 10.0,
	"core_dilate_iterations": 0,
	# Tune this to control how far we repair residual bright wings around bad cores.
	"wing_iterations": 1,
	"repair_max_radius": 3,
	"repair_min_neighbors": 4,
	"repair_method": "local_plane",
	"repair_sigma_clip": 3.5,
	"edge_n_spatial_rows": 3,
	"edge_spatial_low": 0,
	"edge_spatial_high": 79,
	"edge_dispersion_low": 0,
	"edge_dispersion_high": 399,
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

print(
	f"Bad clump pixels from ramp diagnostics: {bad_clump_info['n_clump_pixels']} "
	f"({100.0 * bad_clump_info['clump_fraction']:.4f}%)"
)
# %%
# Expand clump cores by one pixel to capture bright residual wings around defects.
repair_mask = pandora.dilate_binary_mask(
	bad_clump_mask,
	iterations=badpix_params["wing_iterations"],
)
wing_only_mask = repair_mask & (~bad_clump_mask)
print(
	f"Expanded repair mask pixels (core+wings): {int(np.sum(repair_mask))} "
	f"({100.0 * np.mean(repair_mask):.4f}%)"
)
# %%
# Build user-facing pixel masks/maps for QA and downstream use.
corrected_pixel_mask = repair_mask.copy()
pixel_status_map = np.zeros(repair_mask.shape, dtype=np.uint8)
hot_only_mask = hot_pixel_mask & (~corrected_pixel_mask)
pixel_status_map[hot_only_mask] = 1
pixel_status_map[bad_clump_mask] = 2
pixel_status_map[wing_only_mask] = 3

print(
	"Pixel status counts [0=good, 1=hot-only, 2=core, 3=wing]:",
	{int(k): int(v) for k, v in zip(*np.unique(pixel_status_map, return_counts=True))},
)
print(
	f"Corrected pixel mask size: {int(np.sum(corrected_pixel_mask))} "
	f"({100.0 * np.mean(corrected_pixel_mask):.4f}%)"
)

# Quick-look map so corrected pixels are easy to inspect.
fig, ax = plt.subplots(figsize=(8, 6))
cmap = ListedColormap(["black", "gold", "red", "cyan"])
im = ax.imshow(np.rot90(pixel_status_map), origin="lower", cmap=cmap, vmin=0, vmax=3, aspect="auto")
ax.set_title("Pixel Status Map (Core/Wing Classification)")
ax.set_xlabel("Dispersion pixel")
ax.set_ylabel("Spatial pixel")
cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
cb.ax.set_yticklabels(["good", "hot-only", "core", "wing"])
plt.tight_layout()
plt.savefig('images/pixel_status_map.png')
plt.show()
# %%
# === RESIDUAL RAMP ANALYSIS ===
# Analyze residuals to identify non-linearity patterns
# This documents the ~0.3% systematic non-linearity present in the detector
# (Not corrected, as empirical correction degraded photometric precision)

# Combine all bad pixels: hot + repair_mask
all_bad_pixels = hot_pixel_mask | repair_mask

print("\n" + "="*60)
print("MEDIAN RESIDUAL RAMP ANALYSIS")
print("="*60)
print("Note: ~0.3% non-linearity documented but not corrected")
print("Systematic pattern cancels in differential photometry")
print("="*60)

# Analyze residuals for ALL good pixels (excluding bad/hot/edge)
print("\n" + "="*60)
print("ALL GOOD PIXELS")
print("="*60)
residuals_all = pandora.analyze_median_residual_ramp(
    ramp_cube, slope_cube, intercept_cube, exposure_time_s[0:6],
    bad_pixel_mask=all_bad_pixels,
    exclude_edge_rows=3,
    exclude_edge_cols=10,
    title_suffix=" - All Good Pixels"
)

# %%
# Analyze residuals for BRIGHT pixels (top 10% - likely spectral trace)
print("\n" + "="*60)
print("BRIGHT PIXELS (Top 10% flux)")
print("="*60)
residuals_bright = pandora.analyze_median_residual_ramp(
    ramp_cube, slope_cube, intercept_cube, exposure_time_s[0:6],
    bad_pixel_mask=all_bad_pixels,
    flux_percentile_low=90,
    exclude_edge_rows=3,
    exclude_edge_cols=10,
    title_suffix=" - Bright Pixels"
)

# %%
# Analyze residuals for DIM pixels (bottom 50% - likely background)
print("\n" + "="*60)
print("DIM PIXELS (Bottom 50% flux)")
print("="*60)
residuals_dim = pandora.analyze_median_residual_ramp(
    ramp_cube, slope_cube, intercept_cube, exposure_time_s[0:6],
    bad_pixel_mask=all_bad_pixels,
    flux_percentile_high=50,
    exclude_edge_rows=3,
    exclude_edge_cols=10,
    title_suffix=" - Dim Pixels"
)

# %%
# Compare bright vs dim residuals
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

times = exposure_time_s[0:6]

# Top: Absolute residuals
ax = axes[0]
ax.plot(times, residuals_bright['median_residual_dn'], 'o-', 
        markersize=8, linewidth=2, color='tab:red', label='Bright (top 10%)')
ax.plot(times, residuals_dim['median_residual_dn'], 's-', 
        markersize=8, linewidth=2, color='tab:blue', label='Dim (bottom 50%)')
ax.plot(times, residuals_all['median_residual_dn'], '^-', 
        markersize=6, linewidth=1.5, color='gray', alpha=0.6, label='All good pixels')
ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('Median Residual (DN)', fontsize=11)
ax.set_title('Bright vs Dim Pixel Residuals\\n(~0.3% systematic non-linearity documented)', 
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Bottom: Percentage residuals
ax = axes[1]
ax.plot(times, residuals_bright['median_residual_pct'], 'o-', 
        markersize=8, linewidth=2, color='tab:red', label='Bright (top 10%)')
ax.plot(times, residuals_dim['median_residual_pct'], 's-', 
        markersize=8, linewidth=2, color='tab:blue', label='Dim (bottom 50%)')
ax.plot(times, residuals_all['median_residual_pct'], '^-', 
        markersize=6, linewidth=1.5, color='gray', alpha=0.6, label='All good pixels')
ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Median Residual (%)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/residuals_bright_vs_dim_comparison.png', dpi=150)
plt.show()

# Print comparison statistics
print("\n" + "="*60)
print("BRIGHT vs DIM RESIDUAL COMPARISON")
print("="*60)
print(f"\nFirst group residual (t={times[0]:.2f}s):")
print(f"  Bright: {residuals_bright['median_residual_dn'][0]:+.2f} DN "
      f"({residuals_bright['median_residual_pct'][0]:+.3f}%)")
print(f"  Dim:    {residuals_dim['median_residual_dn'][0]:+.2f} DN "
      f"({residuals_dim['median_residual_pct'][0]:+.3f}%)")
print(f"  Difference: {residuals_bright['median_residual_dn'][0] - residuals_dim['median_residual_dn'][0]:+.2f} DN")

print(f"\nLast group residual (t={times[-1]:.2f}s):")
print(f"  Bright: {residuals_bright['median_residual_dn'][-1]:+.2f} DN "
      f"({residuals_bright['median_residual_pct'][-1]:+.3f}%)")
print(f"  Dim:    {residuals_dim['median_residual_dn'][-1]:+.2f} DN "
      f"({residuals_dim['median_residual_pct'][-1]:+.3f}%)")
print(f"  Difference: {residuals_bright['median_residual_dn'][-1] - residuals_dim['median_residual_dn'][-1]:+.2f} DN")

print("\n" + "="*60)
print("CONCLUSION: ~0.3% flux-dependent non-linearity detected")
print("Pattern is systematic and cancels in differential photometry")
print("Empirical correction tested but degraded precision (not applied)")
print("="*60 + "\n")

print("\n" + "="*60)
print("END RESIDUAL ANALYSIS")
print("="*60 + "\n")

# %%
slope_cube_repaired, repair_info = pandora.correct_bad_pixels_with_neighbors(
	slope_cube,
	bad_mask=repair_mask,
	max_radius=badpix_params["repair_max_radius"],
	min_neighbors=badpix_params["repair_min_neighbors"],
	method=badpix_params["repair_method"],
	sigma_clip=badpix_params["repair_sigma_clip"],
)

print(
	f"Neighbor-repaired bad pixels in slope cube: {repair_info['n_fixed']} "
	f"({100.0 * repair_info['fixed_fraction']:.3f}% of masked samples)"
)

slope_cube_repaired, edge_info = pandora.correct_detector_edge_pixels(
	slope_cube_repaired,
	n_spatial_edge_rows=badpix_params["edge_n_spatial_rows"],
	spatial_low=badpix_params["edge_spatial_low"],
	spatial_high=badpix_params["edge_spatial_high"],
	dispersion_low=badpix_params["edge_dispersion_low"],
	dispersion_high=badpix_params["edge_dispersion_high"],
)

print(
	f"Edge correction applied: total={edge_info['n_replaced_total']} samples "
	f"(spatial-edge median={edge_info['n_replaced_spatial_median']}, dispersion-edge mean={edge_info['n_replaced_dispersion_mean']})"
)
# %%
cr_params = {
	"window_frames": 9,
	"sigma": 5.0,  # Increased from 4.5 - less aggressive after timing fix  
	"min_neighbors": 3,
	"positive_only": True,
	"plot_counts_vs_integration": True,
}

slope_cube_cr_corrected, cr_mask, cr_info = pandora.correct_cosmic_rays_temporally(
	slope_cube_repaired,
	window_frames=cr_params["window_frames"],
	sigma=cr_params["sigma"],
	min_neighbors=cr_params["min_neighbors"],
	positive_only=cr_params["positive_only"],
)

print(
	f"Cosmic-ray correction applied: replaced {cr_info['n_replaced']} samples "
	f"({100.0 * cr_info['replacement_fraction']:.4f}% of the cube)"
)

if cr_params["plot_counts_vs_integration"]:
	cr_stats = pandora.plot_cosmic_ray_counts_by_integration(cr_mask)
	print(
		"CR counts per integration: "
		f"median={cr_stats['median']:.1f}, p95={cr_stats['p95']:.1f}, "
		f"max={cr_stats['max']} at integration {cr_stats['argmax']}"
	)
# %%
pandora.display_correction_comparison(
	slope_cube_repaired,
	slope_cube_cr_corrected,
	image_index=44,
	scale_style="zscale",
    iraf_contrast=0.1
)
# %%
pandora.display_science_image(slope_cube_cr_corrected, image_index=44, scale_style="zscale", vmin=100, vmax=200)
# %%
# Tunable trace/aperture priors.
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
# %%
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

print(
	"Variable aperture bounds: "
	f"left {trace_params['spatial_left_start']:.1f}->{trace_params['spatial_left_end']:.1f}, "
	f"right {trace_params['spatial_right_start']:.1f}->{trace_params['spatial_right_end']:.1f}"
)
print(f"Detected peak positions: {trace_est['peak_positions']}")
print(f"Photometric aperture pixels: {int(np.sum(aperture_model['aperture_mask']))}")

pandora.plot_spatial_profile(trace_est, trace_params)
pandora.plot_aperture_overlay(slope_cube_cr_corrected, aperture_model, trace_est,image_index=44)
plt.savefig('images/aperture_overlay.png')
# %%
# Spectrophotometric extraction parameters.
extract_params = {
    "subtract_local_background": True,
    "background_inner_gap": 2,
    "background_width": 10,
    "use_wavelength_weights": True,   # weight white-light by flux per wavelength (S/N weighting)
    "transit_ingress_index": 200,    # first integration inside transit (corrected from ephemeris: actual ~152, set to 0 if no transit)
    "transit_egress_index": 315,     # last integration inside transit (corrected from ephemeris: actual ~299, set to nint-1 if no transit)
    "motion_aperture_guard_pixels": 2,
    "flux_window_low": 0.97,
    "flux_window_high": 1.03,
    "excursion_sigma": 6.0,
    "excursion_padding": 1,
    "motion_sigma": 6.0,
}
# %%
# Perform spectrophotometric extraction.
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
n_bad_channels = int(np.sum(~channel_good))
print(
    f"Channel quality mask: {n_bad_channels} bad channels out of "
    f"{channel_good.size} ({100.0 * n_bad_channels / channel_good.size:.1f}%) "
    f"at dispersion pixels: {extracted_dispersion[~channel_good].tolist()}"
)
print(
    f"Photometry bad-pixel mask: {int(np.sum(photometry_bad_mask))} pixels flagged "
    f"({100.0 * np.mean(photometry_bad_mask):.4f}%)"
)

extracted_spectra_masked = extracted_spectra.copy()
extracted_spectra_masked[:, ~channel_good] = np.nan

wl = pandora.compute_white_light_products(
    extracted_spectra_masked, extraction_info, exposure_time_s, nint, ngroup,
    transit_ingress_index=extract_params["transit_ingress_index"],
    transit_egress_index=extract_params["transit_egress_index"],
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

# --- Gap / reacquisition masking ---
# Discard the first N integrations at the very start of the observation.
integration_keep_mask[:gap_params["initial_frames"]] = False

# Find time jumps that indicate an Earth-occultation or SAA gap.
_dt_s = np.diff(integration_jd) * 86400.0
_gap_starts = np.where(_dt_s > gap_params["gap_threshold_s"])[0]
print(f"Observation gaps detected at integration indices: {_gap_starts.tolist()}")
for _g in _gap_starts:
	_lo = int(_g) + 1
	_hi = min(nint, int(_g) + 1 + gap_params["reacq_frames"])
	integration_keep_mask[_lo:_hi] = False

if gap_params["manual_bad_indices"]:
	for _idx in gap_params["manual_bad_indices"]:
		if 0 <= int(_idx) < nint:
			integration_keep_mask[int(_idx)] = False
	print(f"Manual bad integrations excluded: {gap_params['manual_bad_indices']}")

white_light_clean = white_light_norm.copy()
white_light_clean[~integration_keep_mask] = np.nan

noise_budget = pandora.compute_noise_budget_summary(
	white_light_clean,
	oot_mask,
	median_background_per_pixel,
	integration_time_axis=integration_time_axis,
	initial_frames=gap_params["initial_frames"],
	burn_in_consecutive_frames=5,
)

print(
    f"Clean white-light: kept {int(np.sum(integration_keep_mask))}/{integration_keep_mask.size} integrations, "
    f"excluded {int(np.sum(~integration_keep_mask))} excursions."
)
print(f"Extracted spectra shape: {extracted_spectra_masked.shape}  (integration × dispersion)")
print(f"White-light stats: median={np.nanmedian(wl['white_light']):.6g}, std={np.nanstd(wl['white_light']):.6g}")
print(f"Pointing drift: dx rms={np.nanstd(dx):.4f} pix, dy rms={np.nanstd(dy):.4f} pix")
print(f"OOT white-light RMS: {noise_budget['oot_rms_ppm']:.1f} ppm (MAD-equivalent {noise_budget['oot_mad_ppm']:.1f} ppm)")
if noise_budget["background_rate_per_pixel"] is not None:
	print(
		"Background: "
		f"median={np.nanmedian(noise_budget['background_per_pixel']):.6g} counts/pix/integration, "
		f"rate={np.nanmedian(noise_budget['background_rate_per_pixel']):.6g} counts/pix/s"
	)
else:
	print(
		"Background: "
		f"median={np.nanmedian(noise_budget['background_per_pixel']):.6g} counts/pix/integration"
	)
if noise_budget["burn_in_time_s"] is not None:
	print(
		"Burn-in estimate: "
		f"{noise_budget['burn_in_frames']} frames (~{noise_budget['burn_in_time_s']:.2f} s) "
		f"to settle within {1.0e6 * noise_budget['burn_threshold']:.1f} ppm"
	)
	print(f"Persistence excess in first integrations: {noise_budget['persistence_excess_ppm']:.1f} ppm")
else:
	print("Burn-in estimate: insufficient stable out-of-transit baseline to measure settling time")

pandora.plot_spectrophotometry_diagnostics(
    extracted_dispersion, median_spectrum, channel_good,
    integration_time_axis, white_light_norm, white_light_clean,
    median_background_per_pixel, normalized_spectra,
    dx, oot_mask, spectral_scatter_ppm, time_axis_label,
)
pandora.plot_flux_motion_correlation(dx, dy, white_light_norm, integration_time_axis, time_axis_label)
# %%
# === PHOTOMETRIC ERROR BUDGET ANALYSIS ===
# Compute expected noise from read noise, star shot noise, and background shot noise
# Compare to measured scatter to identify excess noise from systematics

# Add group times to extraction_info
extraction_info['group_times'] = exposure_time_s[0:6]

error_budget = pandora.compute_photometric_error_budget(
    white_light=wl['white_light'],
    white_light_norm=white_light_norm,
    oot_mask=oot_mask,
    extraction_info=extraction_info,
    median_background_per_pixel=median_background_per_pixel,
    integration_time_axis=integration_time_axis,
    read_noise_e=READ_NOISE,
    gain_e_per_dn=GAIN,
)
# %%
# PCA telemetry decorrelation parameters.
pca_params = {
	"telemetry_hdu_names": ("VITL_DATA", "SC_QUATERNIONS", "SC_POSITION", "SC_VELOCITY"),
	"n_components": 3,
	# Polynomial order in time used alongside PCA + motion + background regressors.
	"time_poly_order": 2,
}

# Transit/event ephemeris.
ephem_params = {
	"period_days": 3.3448285,
	"t0_hjd_utc": 2456927.06839,
	"t14_hours": 3.470,
}

# Physical transit model parameters for WASP-178b.
# Limb darkening u1/u2 for ~1.2 μm, Teff≈9350K (A1V), logg≈4.0 (Claret 2017 near-IR).
transit_params = {
	"a_over_Rstar": 7.17,       # Semi-major axis / stellar radius
	"b": 0.54,                  # Impact parameter
	"rp_over_Rstar": 0.11066,   # Planet radius / stellar radius
	"u1": 0.10,                 # Quadratic LD u1 (~1.2 μm, A1V host)
	"u2": 0.20,                 # Quadratic LD u2
}
# %%
# --- PCA telemetry decorrelation ---
# Compute predicted event window from ephemeris (used in the plot below).
_epoch = int(np.rint((np.nanmedian(integration_jd) - ephem_params["t0_hjd_utc"]) / ephem_params["period_days"]))
event_center_jd   = ephem_params["t0_hjd_utc"] + _epoch * ephem_params["period_days"]
event_ingress_jd  = event_center_jd - 0.5 * ephem_params["t14_hours"] / 24.0
event_egress_jd   = event_center_jd + 0.5 * ephem_params["t14_hours"] / 24.0

# Build a telemetry matrix from the four spacecraft HDUs, standardise, and
# compute PCA components via SVD.  A linear systematic model is then fit
# OOT-only and removed from the full time series.
_tel_series = pandora.sample_fits_telemetry_to_integrations(
	fits_path,
	integration_jd,
	hdu_names=pca_params["telemetry_hdu_names"],
)

_tel_cols = []
for _k, _v in _tel_series.items():
	if np.all(np.isfinite(_v)) and np.nanstd(_v) > 0:
		_tel_cols.append((_v - np.nanmean(_v)) / np.nanstd(_v))

if len(_tel_cols) == 0:
	print("WARNING: No usable telemetry columns found for PCA decorrelation.")
	white_light_pca = white_light_clean.copy()
else:
	_tel_matrix = np.column_stack(_tel_cols)
	_u, _s_svd, _vh = np.linalg.svd(_tel_matrix, full_matrices=False)
	_n_comp = min(pca_params["n_components"], _u.shape[1])
	_pca_comps = _u[:, :_n_comp]
	print(f"PCA: using {_n_comp} components from {len(_tel_cols)} telemetry channels.")

	# --- PCA Diagnostic Plots ---
	# 1. Scree plot: Explained variance ratio to validate number of components
	_explained_variance = (_s_svd ** 2) / np.sum(_s_svd ** 2)
	_cumulative_variance = np.cumsum(_explained_variance)
	
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
	
	# Scree plot
	n_show = min(15, len(_explained_variance))  # Show up to 15 components
	ax1.plot(np.arange(1, n_show + 1), 100 * _explained_variance[:n_show], 
	         'o-', color='tab:blue', linewidth=2, markersize=6)
	ax1.axvline(_n_comp, color='tab:red', linestyle='--', linewidth=2, 
	            label=f'Selected: {_n_comp} components')
	ax1.set_xlabel('Principal Component', fontsize=11)
	ax1.set_ylabel('Explained Variance (%)', fontsize=11)
	ax1.set_title('Scree Plot: Individual Explained Variance', fontsize=12, fontweight='bold')
	ax1.grid(True, alpha=0.3)
	ax1.legend(fontsize=9)
	ax1.set_xticks(np.arange(1, n_show + 1))
	
	# Cumulative variance
	ax2.plot(np.arange(1, n_show + 1), 100 * _cumulative_variance[:n_show], 
	         's-', color='tab:green', linewidth=2, markersize=6)
	ax2.axvline(_n_comp, color='tab:red', linestyle='--', linewidth=2, 
	            label=f'{_n_comp} comp: {100*_cumulative_variance[_n_comp-1]:.1f}%')
	ax2.axhline(100 * _cumulative_variance[_n_comp - 1], color='tab:red', 
	            linestyle=':', linewidth=1, alpha=0.5)
	ax2.set_xlabel('Number of Components', fontsize=11)
	ax2.set_ylabel('Cumulative Variance (%)', fontsize=11)
	ax2.set_title('Cumulative Explained Variance', fontsize=12, fontweight='bold')
	ax2.grid(True, alpha=0.3)
	ax2.legend(fontsize=9)
	ax2.set_xticks(np.arange(1, n_show + 1))
	ax2.set_ylim(0, 105)
	
	plt.tight_layout()
	plt.savefig('images/pca_variance_explained.png', dpi=150)
	plt.show()
	
	print(f"First {_n_comp} components explain {100*_cumulative_variance[_n_comp-1]:.2f}% of variance")
	for i in range(min(_n_comp, len(_explained_variance))):
		print(f"  PC{i+1}: {100*_explained_variance[i]:.2f}%")
	
	# 2. Principal components time series plot
	fig, axes = plt.subplots(_n_comp, 1, figsize=(11, 3 * _n_comp), sharex=True)
	if _n_comp == 1:
		axes = [axes]
	
	_t_plot_jd = integration_jd
	_valid_plot = integration_keep_mask.copy()
	
	for i in range(_n_comp):
		ax = axes[i]
		# Plot all points in gray, valid points in color
		ax.plot(_t_plot_jd, _pca_comps[:, i], 'o', color='0.7', markersize=3, alpha=0.5)
		ax.plot(_t_plot_jd[_valid_plot], _pca_comps[_valid_plot, i], 'o', 
		        color=f'C{i}', markersize=4, label=f'PC{i+1}')
		
		# Mark transit window
		ax.axvspan(event_ingress_jd, event_egress_jd, color='tab:blue', alpha=0.1, 
		           label='Predicted transit' if i == 0 else '')
		ax.axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)
		
		ax.set_ylabel(f'PC{i+1}\n({100*_explained_variance[i]:.1f}% var)', fontsize=10)
		ax.legend(loc='upper right', fontsize=8)
		ax.grid(True, alpha=0.3)
		
		# Add correlation with white light if available
		if i == 0 and 'white_light_norm' in dir():
			_corr_valid = _valid_plot & np.isfinite(white_light_norm)
			if np.sum(_corr_valid) > 10:
				_corr = np.corrcoef(_pca_comps[_corr_valid, i], 
				                   white_light_norm[_corr_valid])[0, 1]
				ax.text(0.02, 0.95, f'Corr w/ WL: {_corr:.3f}', 
				        transform=ax.transAxes, fontsize=8,
				        verticalalignment='top', bbox=dict(boxstyle='round', 
				        facecolor='wheat', alpha=0.5))
	
	axes[-1].set_xlabel('JD', fontsize=11)
	fig.suptitle('PCA Principal Components vs Time', fontsize=13, fontweight='bold', y=0.995)
	plt.tight_layout()
	plt.savefig('images/pca_components_timeseries.png', dpi=150)
	plt.show()

	# Build design matrix for the kept integrations only.
	_valid = integration_keep_mask.copy()
	_t_rel = integration_jd - np.nanmedian(integration_jd[_valid])
	_bg_per_int = np.nanmedian(extraction_info["background_per_pixel"], axis=1)

	_poly_cols = [_t_rel ** p for p in range(pca_params["time_poly_order"] + 1)]
	_X_full = np.column_stack(_poly_cols + [dx, dy, _bg_per_int] + [_pca_comps[:, c] for c in range(_n_comp)])

	_X_oot = _X_full[_valid & oot_mask]
	_y_oot = white_light_norm[_valid & oot_mask]

	if _X_oot.shape[0] < _X_oot.shape[1] + 1:
		print("WARNING: Too few OOT integrations for PCA fit; skipping decorrelation.")
		white_light_pca = white_light_clean.copy()
	else:
		_c_pca, _, _, _ = np.linalg.lstsq(_X_oot, _y_oot, rcond=None)
		_baseline_full = _X_full @ _c_pca

		_wl_valid = white_light_norm[_valid]
		_base_valid = _baseline_full[_valid]
		_oot_valid = oot_mask[_valid]

		_detrended = _wl_valid / _base_valid
		_detrended /= np.nanmedian(_detrended[_oot_valid])

		white_light_pca = np.full(nint, np.nan)
		white_light_pca[_valid] = _detrended

		pca_oot_scatter_ppm = 1.0e6 * np.nanstd(_detrended[_oot_valid])
		raw_oot_scatter_ppm = 1.0e6 * np.nanstd(_wl_valid[_oot_valid])
		pca_reduction_pct = 100.0 * (1.0 - (pca_oot_scatter_ppm / raw_oot_scatter_ppm)) if raw_oot_scatter_ppm > 0 else np.nan
		print(f"OOT scatter before PCA detrending: {raw_oot_scatter_ppm:.1f} ppm")
		print(f"OOT scatter  after PCA detrending: {pca_oot_scatter_ppm:.1f} ppm")
		print(f"PCA detrending reduced OOT scatter by {pca_reduction_pct:.1f}%")

		# Diagnostic plot: raw + systematic model + detrended (offset -0.05).
		_t_plot = integration_jd[_valid]
		fig, ax = plt.subplots(figsize=(11, 4.5))
		ax.scatter(_t_plot, _wl_valid, color="0.6", s=6, label="Raw normalised")
		ax.plot(_t_plot, _base_valid, color="tab:red", lw=1.2, alpha=0.7, label="Systematic model")
		ax.scatter(_t_plot, _detrended - 0.05, color="tab:green", s=6, label="Detrended (−0.05 offset)")
		ax.axhline(1.0, color="k", ls=":", lw=1)
		ax.axhline(0.95, color="k", ls=":", lw=1)
		ax.axvline(event_center_jd, color="tab:blue", lw=1.2, ls="--", label="Predicted mid-transit")
		ax.axvspan(event_ingress_jd, event_egress_jd, color="tab:blue", alpha=0.14, label="Predicted T14")
		ax.set_xlabel("JD")
		ax.set_ylabel("Normalized white-light flux")
		ax.set_ylim(0.92, 1.05)
		ax.set_title("WASP-178b — PCA-detrended white-light curve")
		ax.legend(loc="best", fontsize=8)
		plt.tight_layout()
		plt.savefig('images/pca_lightcurve.png')
		plt.show()
# %%
# Transit model overlay on PCA-detrended white-light curve (WASP-178b).
import pytfit5.transitmodel as tm

# Derive stellar density (g/cm³) from a/R* and period using Kepler's 3rd law.
_a_rs = transit_params["a_over_Rstar"]
_per_s = ephem_params["period_days"] * 86400.0
_rho_gcc = (_a_rs**3 * 3.0 * np.pi
		   / (1000.0 * 6.674e-11 * _per_s**2))

# Convert standard u1/u2 to Kipping (2013) q1/q2 parameterisation used by pytfit5.
_u1, _u2 = transit_params["u1"], transit_params["u2"]
_q1 = (_u1 + _u2) ** 2
_q2 = _u1 / (2.0 * (_u1 + _u2))

# Build transit model solution object.
sol_transit = tm.transit_model_class()
sol_transit.npl   = 1
sol_transit.rho   = _rho_gcc
sol_transit.nl3   = _q1           # Kipping q1
sol_transit.nl4   = _q2           # Kipping q2
sol_transit.t0    = [event_center_jd]
sol_transit.per   = [ephem_params["period_days"]]
sol_transit.bb    = [transit_params["b"]]
sol_transit.rdr   = [transit_params["rp_over_Rstar"]]
sol_transit.ecw   = [0.0]
sol_transit.esw   = [0.0]
sol_transit.zpt   = 0.0
sol_transit.dil   = 0.0

print(
	f"Transit model: ρ*={_rho_gcc:.4f} g/cm³, a/R*={_a_rs:.2f}, "
	f"depth={(transit_params['rp_over_Rstar']**2)*1e6:.0f} ppm"
)

# Integration time per point (days); fall back to 2-min default if unavailable.
_itime_days = (
	np.asarray(exposure_time_s, dtype=float) / 86400.0
	if exposure_time_s is not None
	else np.full(len(integration_jd), 2.0 / 1440.0)
)

# Evaluate model at each data timestamp.
_model_data = tm.transitModel(sol_transit, integration_jd, _itime_days, nintg=41)

# Fine-sampled model curve for a smooth overlay.
_jd_fine = np.linspace(integration_jd.min(), integration_jd.max(), 3000)
_itime_fine = np.full(3000, float(np.nanmedian(_itime_days)))
_model_fine = tm.transitModel(sol_transit, _jd_fine, _itime_fine, nintg=41)

# Residuals on kept integrations.
_keep = integration_keep_mask
_resid_pca = white_light_pca[_keep] - _model_data[_keep]
_oot_keep = oot_mask[_keep]
_resid_rms_ppm = 1.0e6 * np.nanstd(_resid_pca[_oot_keep])

fig, axes = plt.subplots(
	2, 1, figsize=(11, 7), sharex=True,
	gridspec_kw={"height_ratios": [3, 1]},
)
ax0 = axes[0]
ax0.scatter(integration_jd, white_light_pca, color="tab:orange", s=7,
			label="PCA-detrended", zorder=2)
ax0.plot(_jd_fine, _model_fine, color="tab:red", lw=1.5,
		 label="Transit model", zorder=3)
ax0.axhline(1.0, color="k", ls=":", lw=0.8)
ax0.axvline(event_center_jd, color="tab:blue", lw=1.0, ls="--", alpha=0.6)
ax0.axvspan(event_ingress_jd, event_egress_jd, color="tab:blue", alpha=0.10,
			 label="Predicted T14")
ax0.set_ylabel("Normalized flux")
ax0.set_ylim(0.97, 1.03)
ax0.set_title(
	f"WASP-178b  |  a/R*={_a_rs:.2f}, b={transit_params['b']:.2f}, "
	f"Rp/R*={transit_params['rp_over_Rstar']:.5f}, ρ*={_rho_gcc:.3f} g/cm³"
)
ax0.legend(loc="best", fontsize=8)

ax1 = axes[1]
ax1.scatter(
	integration_jd[_keep], _resid_pca * 1.0e6,
	color="tab:orange", s=5, label="Residual",
)
ax1.axhline(0.0, color="k", ls=":", lw=0.8)
ax1.set_ylabel("Residual (ppm)")
ax1.set_xlabel("JD")
ax1.set_title(f"OOT residuals RMS = {_resid_rms_ppm:.1f} ppm")
plt.tight_layout()
plt.savefig('images/transit_model_fit.png')
plt.show()
# %%
