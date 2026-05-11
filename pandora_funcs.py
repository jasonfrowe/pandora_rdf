import numpy as np
# from pathlib import Path
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from numba import njit, prange


def read_rdf_raw_science(fits_path):
	"""Read the RAW SCIENCE array from an RDF FITS file in native dimensions."""
	with fits.open(fits_path, memmap=False) as hdul:
		if "RAW SCIENCE" not in hdul:
			raise KeyError("RAW SCIENCE extension not found in RDF FITS file.")
		raw = np.asarray(hdul["RAW SCIENCE"].data)
		header = hdul["RAW SCIENCE"].header.copy()

	if raw.ndim != 4:
		raise ValueError(
			f"RAW SCIENCE is expected to be 4D (integration, group, row, column); got {raw.shape}."
		)

	return raw, header

def flatten_ramp_cube(ramp_cube):
	"""Flatten a 4D ramp cube (integration, group, row, column) into 3D frames."""
	ramp = np.asarray(ramp_cube)
	if ramp.ndim != 4:
		raise ValueError(f"ramp_cube must be 4D, got shape {ramp.shape}.")
	nint, ngroup, ny, nx = ramp.shape
	return ramp.reshape(nint * ngroup, ny, nx)


def unflatten_ramp_cube(flat_cube, nint, ngroup):
	"""Restore a flattened 3D cube back to 4D ramp shape."""
	flat = np.asarray(flat_cube)
	if flat.ndim != 3:
		raise ValueError(f"flat_cube must be 3D, got shape {flat.shape}.")
	if flat.shape[0] != nint * ngroup:
		raise ValueError(
			f"Frame axis {flat.shape[0]} does not match nint*ngroup={nint * ngroup}."
		)
	return flat.reshape(nint, ngroup, flat.shape[1], flat.shape[2])

def get_rdf_auxiliary_data(fits_path):
	"""Read ROW/COLUMN maps and per-frame exposure times from an RDF FITS file."""
	with fits.open(fits_path, memmap=False) as hdul:
		row_map = np.asarray(hdul["ROW"].data) if "ROW" in hdul else None
		col_map = np.asarray(hdul["COLUMN"].data) if "COLUMN" in hdul else None
		exp_time = np.asarray(hdul["EXPOSURE_TIME"].data) if "EXPOSURE_TIME" in hdul else None

	if exp_time is not None:
		exp_time = np.asarray(exp_time, dtype=float).reshape(-1)

	return row_map, col_map, exp_time

def read_rdf_time_extension(fits_path):
	"""Read the TIME extension as (integration, group) array."""
	with fits.open(fits_path, memmap=False) as hdul:
		if "TIME" not in hdul:
			raise KeyError("TIME extension not found in RDF FITS file.")
		time_arr = np.asarray(hdul["TIME"].data, dtype=float)

	if time_arr.ndim != 2:
		raise ValueError(f"TIME extension must be 2D, got shape {time_arr.shape}.")

	return time_arr

def display_science_image(
	cube,
	image_index=0,
	scale_style="zscale",
	cmap="viridis",
	vmin=None,
	vmax=None,
	iraf_contrast=0.25,
):
	"""Display one image from a SCIENCE datacube.

	Parameters
	----------
	cube : numpy.ndarray
		3D SCIENCE datacube with shape (n_frames, ny, nx).
	image_index : int, optional
		Index of the image/frame to display. Default is 0.
	scale_style : str, optional
		Display scale style: "none", "zscale", "log", or "sqrt".
		Default is "zscale".
	cmap : str, optional
		Matplotlib colormap name. Default is "viridis".
	vmin, vmax : float, optional
		Display scaling limits. User-provided values are honored for all
		scale styles.
	iraf_contrast : float, optional
		Contrast parameter for IRAF-style zscale. Default is 0.25.
	"""
	if cube.ndim != 3:
		raise ValueError(f"Expected a 3D datacube; got array with shape {cube.shape}.")

	if not 0 <= image_index < cube.shape[0]:
		raise IndexError(
			f"image_index={image_index} is out of bounds for {cube.shape[0]} frames."
		)

	frame = cube[image_index]
	finite = np.isfinite(frame)
	if not np.any(finite):
		raise ValueError("Selected frame has no finite values for display scaling.")

	user_set_vmin = vmin is not None
	user_set_vmax = vmax is not None

	valid_styles = {"none", "zscale", "log", "sqrt"}
	style = str(scale_style).lower()
	if style not in valid_styles:
		raise ValueError(
			f"scale_style must be one of {sorted(valid_styles)}; got '{scale_style}'."
		)

	data_min = float(np.nanmin(frame[finite]))
	data_max = float(np.nanmax(frame[finite]))

	if style == "zscale":
		zscale = ZScaleInterval(contrast=iraf_contrast)
		auto_vmin, auto_vmax = zscale.get_limits(frame[finite])
	elif style == "log":
		positive = frame[finite & (frame > 0)]
		if positive.size == 0:
			raise ValueError(
				"Log scaling requires at least one positive pixel in the selected frame."
			)

		# Use robust defaults so log mode is useful without manual limits.
		auto_vmin = float(np.nanpercentile(positive, 1.0))
		auto_vmax = float(np.nanpercentile(positive, 99.5))
		if auto_vmin <= 0 or auto_vmin >= auto_vmax:
			auto_vmin = float(np.nanmin(positive))
			auto_vmax = float(np.nanmax(positive))
	else:
		auto_vmin, auto_vmax = data_min, data_max

	if vmin is None:
		vmin = auto_vmin
	if vmax is None:
		vmax = auto_vmax

	if style in {"none", "zscale"}:
		if vmin >= vmax:
			raise ValueError(f"vmin ({vmin}) must be less than vmax ({vmax}).")
		norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
	elif style == "sqrt":
		if vmin >= vmax:
			raise ValueError(f"vmin ({vmin}) must be less than vmax ({vmax}).")
		norm = colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax, clip=True)
	else:
		if (vmin <= 0 and user_set_vmin) or (vmax <= 0 and user_set_vmax):
			raise ValueError(
				"Log scaling requires user-provided vmin and vmax to be > 0."
			)
		if vmin <= 0:
			vmin = float(np.nextafter(0, 1))
		if vmax <= 0:
			vmax = float(np.nanmax(frame[finite & (frame > 0)]))
		if vmin >= vmax:
			raise ValueError(f"vmin ({vmin}) must be less than vmax ({vmax}).")
		norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)

	plot_frame = np.rot90(frame)
	ny, nx = plot_frame.shape
	fig_width = 8.0
	fig_height = max(3.5, fig_width * (ny / max(nx, 1)))
	fig, ax = plt.subplots(figsize=(fig_width, fig_height))
	img = ax.imshow(
		plot_frame,
		origin="lower",
		cmap=cmap,
		norm=norm,
		aspect="equal",
		interpolation="nearest",
	)
	ax.set_title(f"SCIENCE image {image_index} ({style})")
	ax.set_xlabel("Dispersion pixel")
	ax.set_ylabel("Spatial pixel")
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="2.5%", pad=0.05)
	fig.colorbar(img, cax=cax, label="Counts")
	plt.tight_layout()
	plt.show()

@njit(parallel=True, fastmath=True)
def get_slope_cube_owls(ramp_cube, times, read_noise=10.0, gain=1.0, threshold=4.0):
    """
    Computes CR-rejected, optimally weighted least squares slopes for a full JWST data cube.
    
    Parameters:
    - ramp_cube: 4D numpy array [nint, ngroup, nx, ny]
    - times: 1D numpy array of exposure times per group
    - read_noise: Detector read noise in electrons (estimate)
    - gain: Detector gain in e-/DN
    - threshold: Sigma threshold for jump detection
    
    Returns:
    - slope_cube: 3D numpy array [nint, nx, ny] containing the calculated slopes
    """
    nint, ngroup, nx, ny = ramp_cube.shape
    slope_cube = np.zeros((nint, nx, ny), dtype=np.float32)
    
    # Pre-calculate delta times
    dt = np.empty(ngroup - 1, dtype=np.float32)
    for g in range(ngroup - 1):
        dt[g] = times[g+1] - times[g]
        
    # prange enables multi-threading on the outermost loop (Integrations)
    for i in prange(nint):
        for x in range(nx):
            for y in range(ny):
                
                # --- 1. Extract 1D Ramp ---
                counts = np.empty(ngroup, dtype=np.float32)
                for g in range(ngroup):
                    counts[g] = ramp_cube[i, g, x, y]
                    
                # --- 2. CR Rejection (Two-point diff & MAD) ---
                rates = np.empty(ngroup - 1, dtype=np.float32)
                for g in range(ngroup - 1):
                    rates[g] = (counts[g+1] - counts[g]) / dt[g]
                    
                # Numba fully supports numpy median operations
                med_rate = np.median(rates)
                
                abs_devs = np.empty(ngroup - 1, dtype=np.float32)
                for g in range(ngroup - 1):
                    abs_devs[g] = np.abs(rates[g] - med_rate)
                mad = np.median(abs_devs)
                
                sigma = mad * 1.4826
                if sigma < 1e-6:
                    # Fallback to standard deviation if the ramp is perfectly noiseless
                    sigma = np.std(rates) + 1e-6
                    
                # Apply Corrections
                corrected_counts = counts.copy()
                for g in range(ngroup - 1):
                    if np.abs(rates[g] - med_rate) > (threshold * sigma):
                        expected_diff = med_rate * dt[g]
                        actual_diff = counts[g+1] - counts[g]
                        excess = actual_diff - expected_diff
                        
                        # Subtract excess from all subsequent groups
                        for k in range(g+1, ngroup):
                            corrected_counts[k] -= excess
                            
                # --- 3. Optimally Weighted Least Squares (OWLS) ---
                sum_w = 0.0
                sum_wx = 0.0
                sum_wy = 0.0
                sum_wxx = 0.0
                sum_wxy = 0.0
                
                for g in range(ngroup):
                    t = times[g]
                    c = corrected_counts[g]
                    
                    # Inverse-variance weighting
                    # Variance = Read Noise^2 + Poisson Noise (Signal / Gain)
                    variance = (read_noise ** 2) + (max(c, 0.0) / gain)
                    w = 1.0 / variance
                    
                    sum_w += w
                    sum_wx += w * t
                    sum_wy += w * c
                    sum_wxx += w * t * t
                    sum_wxy += w * t * c
                    
                delta = (sum_w * sum_wxx) - (sum_wx * sum_wx)
                
                if delta != 0.0:
                    slope_cube[i, x, y] = (sum_w * sum_wxy - sum_wx * sum_wy) / delta
                else:
                    slope_cube[i, x, y] = 0.0
                    
    return slope_cube