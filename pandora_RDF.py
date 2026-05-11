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
}

# Tunable trace/aperture priors.
trace_params = {
	"dispersion_min": 75,
	"dispersion_max": 282,
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
pandora.display_science_image(cube, image_index=20, scale_style="zscale")
# %%
# Typical NIRCam/NIRISS values (Adjust read_noise and gain based on your specific detector)
READ_NOISE = 12.0 # e-
GAIN = 1.5 # e-/DN

print("Compiling and processing...")
start_time = time.time()

times = exposure_time_s[0:ngroup]

slope_cube, intercept_cube, scatter_cube = pandora.get_slope_cube_owls(
	ramp_cube,
	times,
	read_noise=READ_NOISE,
	gain=GAIN,
	threshold=4.0,
	return_diagnostics=True,
)

# %%
pandora.display_science_image(slope_cube, image_index=20, scale_style="zscale",iraf_contrast=0.99)
# %%
bad_mask, bad_info = pandora.detect_bad_pixel_clumps_from_ramp_fit(
	intercept_cube,
	slope_cube,
	scatter_cube=scatter_cube,
	min_intercept_offset_dn=badpix_params["min_intercept_offset_dn"],
	intercept_sigma=badpix_params["intercept_sigma"],
	slope_percentile=badpix_params["slope_percentile"],
	scatter_sigma=badpix_params["scatter_sigma"],
	dilate_iterations=badpix_params["core_dilate_iterations"],
)

repair_mask = pandora.dilate_binary_mask(bad_mask, iterations=badpix_params["wing_iterations"])
slope_cube, repair_info = pandora.correct_bad_pixels_with_neighbors(
	slope_cube,
	repair_mask,
	max_radius=badpix_params["repair_max_radius"],
	min_neighbors=badpix_params["repair_min_neighbors"],
)

print(
	f"Bad-pixel mask: {bad_info['n_clump_pixels']} pixels flagged "
	f"({bad_info['clump_fraction']:.4%})"
)
print(
	f"Repaired {repair_info['n_fixed']} pixel samples in {time.time() - start_time:.3f} seconds."
)
# %%
pandora.display_science_image(slope_cube, image_index=20, scale_style="zscale",iraf_contrast=0.05)
# %%
trace_est = pandora.estimate_trace_aperture(
	slope_cube,
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
	n_spatial=slope_cube.shape[2],
)

print(
	"Variable aperture bounds: "
	f"left {trace_params['spatial_left_start']:.1f}->{trace_params['spatial_left_end']:.1f}, "
	f"right {trace_params['spatial_right_start']:.1f}->{trace_params['spatial_right_end']:.1f}"
)
print(f"Detected peak positions: {trace_est['peak_positions']}")
print(f"Photometric aperture pixels: {int(np.sum(aperture_model['aperture_mask']))}")

pandora.plot_spatial_profile(trace_est, trace_params)
pandora.plot_aperture_overlay(slope_cube, aperture_model, trace_est)
# %%
# Spectrophotometric extraction parameters.
extract_params = {
    "subtract_local_background": True,
    "background_inner_gap": 2,
    "background_width": 10,
    "transit_ingress_index": 200,    # first integration inside transit (set to 0 if no transit)
    "transit_egress_index": 315,     # last integration inside transit  (set to nint-1 if no transit)
    "motion_aperture_guard_pixels": 2,
    "flux_window_low": 0.97,
    "flux_window_high": 1.03,
    "excursion_sigma": 6.0,
    "excursion_padding": 1,
    "motion_sigma": 6.0,
}
# %%
# Perform spectrophotometric extraction.
photometry_bad_mask = bad_mask | repair_mask

extracted_spectra, extracted_dispersion, extraction_info = pandora.extract_trace_spectra_variable_aperture(
    slope_cube,
    aperture_model=aperture_model,
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
    slope_cube, aperture_model,
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

# Difference-imaging motion settings.
diff_motion_params = {
	# Central-difference kernels for image gradients in x (columns) and y (rows).
	"kernel_dx": (-0.5, 0.0, 0.5),
	"kernel_dy": (-0.5, 0.0, 0.5),
	# Fit a variable background plane with offset + x + y terms.
	"background_order": 1,
	# Robust clip in units of MAD-sigma on the difference image residuals.
	"clip_sigma": 8.0,
	# Minimum valid pixels per frame required for a stable shift fit.
	"min_valid_pixels": 2500,
}
# %%
# Explore image drift from difference imaging.
# Build a reference from out-of-transit integrations when available.
if np.any(oot_mask):
	diff_reference = np.nanmedian(slope_cube[oot_mask], axis=0)
else:
	diff_reference = np.nanmedian(slope_cube, axis=0)

diff_shift = pandora.estimate_difference_image_shifts(
	slope_cube,
	reference_image=diff_reference,
	bad_pixel_mask=photometry_bad_mask,
	kernel_dx=diff_motion_params["kernel_dx"],
	kernel_dy=diff_motion_params["kernel_dy"],
	background_order=diff_motion_params["background_order"],
	clip_sigma=diff_motion_params["clip_sigma"],
	min_valid_pixels=diff_motion_params["min_valid_pixels"],
)

dx_diff = diff_shift["dx"]
dy_diff = diff_shift["dy"]
dx_diff -= np.nanmedian(dx_diff[oot_mask])
dy_diff -= np.nanmedian(dy_diff[oot_mask])

valid_motion = np.isfinite(dx_diff) & np.isfinite(dy_diff)
valid_vs_centroid = valid_motion & np.isfinite(dx) & np.isfinite(dy)

print(
	"Difference-imaging fit quality: "
	f"median valid pixels/frame={np.nanmedian(diff_shift['n_valid_pixels']):.0f}, "
	f"median residual RMS={np.nanmedian(diff_shift['residual_rms']):.6g}"
)
print(
	"Difference-imaging drift RMS: "
	f"dx={np.nanstd(dx_diff[valid_motion]):.4f} pix, "
	f"dy={np.nanstd(dy_diff[valid_motion]):.4f} pix"
)
if np.any(valid_vs_centroid):
	corr_dx = np.corrcoef(dx_diff[valid_vs_centroid], dx[valid_vs_centroid])[0, 1]
	corr_dy = np.corrcoef(dy_diff[valid_vs_centroid], dy[valid_vs_centroid])[0, 1]
	print(f"Correlation with aperture-centroid motion: corr(dx)={corr_dx:.3f}, corr(dy)={corr_dy:.3f}")

pandora.plot_difference_imaging_motion(
	integration_time_axis,
	dx_diff,
	dy_diff,
	oot_mask=oot_mask,
	dx_reference=dx,
	dy_reference=dy,
)
# %%
