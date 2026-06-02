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
fig, ax = plot_ramp_at_pixel(ramp_cube, int_idx=44, x=150, y=37, group_times=exposure_time_s[0:6], read_noise=READ_NOISE, gain=GAIN, threshold=4.0)
plt.show()
# %%
