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
# location data products
datadir = '/opt/data2/rowe/pandora/2026/RDF1/data/'
science_file = 'Pandora_RDF_WASP-178b_all.fits'
fits_path = datadir + science_file

# Tunable bad-pixel/clump-repair settings.
badpix_params = {
	"min_intercept_offset_dn": 250.0,
	"intercept_sigma": 10.0,
	"slope_percentile": 0.5,
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
print(
	f"Loaded TIME extension: {time_jd_cube.shape}, "
	f"jd range=({np.nanmin(time_jd_cube):.6f}, {np.nanmax(time_jd_cube):.6f})"
)
# %%
pandora.display_science_image(cube, image_index=269, scale_style="zscale", iraf_contrast=0.99)
# %%
fit_products_pre = pandora.compute_ramp_fit_products_r2s(
	ramp_cube,
	sigcut=2.0,
)
# %%
pandora.display_science_image(fit_products_pre["slope"], image_index=45, \
    scale_style="zscale", iraf_contrast=0.99)

# %%
hot_pixel_mask, hot_info = pandora.detect_hot_pixels_from_ramp_fit(
	fit_products_pre["intercept"],
	scatter_cube=fit_products_pre["scatter"],
	intercept_sigma=8.0,
	min_intercept_offset_dn=150.0,
	scatter_sigma=8.0,
)

print(
	f"Hot pixels from ramp zero-point map: {hot_info['n_hot']} "
	f"({100.0 * hot_info['hot_fraction']:.4f}%)"
)
# %%
bad_clump_mask, bad_clump_info = pandora.detect_bad_pixel_clumps_from_ramp_fit(
	fit_products_pre["intercept"],
	fit_products_pre["slope"],
	scatter_cube=fit_products_pre["scatter"],
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
plt.show()
# %%
slope_cube = fit_products_pre["slope"]
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
pandora.display_correction_comparison(
	fit_products_pre["slope"],
	slope_cube_repaired,
	image_index=45,
	scale_style="zscale",
    iraf_contrast=0.1
)
# %%
trace_est = pandora.estimate_trace_aperture(
	slope_cube_repaired,
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
	n_spatial=slope_cube_repaired.shape[2],
)

print(
	"Variable aperture bounds: "
	f"left {trace_params['spatial_left_start']:.1f}->{trace_params['spatial_left_end']:.1f}, "
	f"right {trace_params['spatial_right_start']:.1f}->{trace_params['spatial_right_end']:.1f}"
)
print(f"Detected peak positions: {trace_est['peak_positions']}")
print(f"Photometric aperture pixels: {int(np.sum(aperture_model['aperture_mask']))}")

pandora.plot_spatial_profile(trace_est, trace_params)
pandora.plot_aperture_overlay(slope_cube_repaired, aperture_model, trace_est)
# %%
# Perform spectrophotometric extraction.
photometry_bad_mask = repair_mask

extracted_spectra, extracted_dispersion, extraction_info = pandora.extract_trace_spectra_variable_aperture(
    slope_cube_repaired,
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
    slope_cube_repaired, aperture_model,
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
white_light_clean = white_light_norm.copy()
white_light_clean[~integration_keep_mask] = np.nan

print(
    f"Clean white-light: kept {int(np.sum(integration_keep_mask))}/{integration_keep_mask.size} integrations, "
    f"excluded {int(np.sum(~integration_keep_mask))} excursions."
)
print(f"Extracted spectra shape: {extracted_spectra_masked.shape}  (integration × dispersion)")
print(f"White-light stats: median={np.nanmedian(wl['white_light']):.6g}, std={np.nanstd(wl['white_light']):.6g}")
print(f"Pointing drift: dx rms={np.nanstd(dx):.4f} pix, dy rms={np.nanstd(dy):.4f} pix")

pandora.plot_spectrophotometry_diagnostics(
    extracted_dispersion, median_spectrum, channel_good,
    integration_time_axis, white_light_norm, white_light_clean,
    median_background_per_pixel, normalized_spectra,
    dx, oot_mask, spectral_scatter_ppm, time_axis_label,
)
pandora.plot_flux_motion_correlation(dx, dy, white_light_norm, integration_time_axis, time_axis_label)
# %%
anim = pandora.animate_datacube(
    slope_cube_repaired,
    title="slope_cube_repaired",
    scale_style="zscale",
    iraf_contrast=0.25,
    interval=80,          # ms per frame
)
# %%
from IPython.display import HTML
HTML(anim.to_jshtml())
# %%
pandora.animate_datacube(
    slope_cube_repaired,
    title="slope_cube_repaired",
    output_path="slope_cube_repaired.mp4",
    fps=15,
    dpi=100,
)
# %%
