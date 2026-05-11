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

def dilate_binary_mask(mask, iterations=1):
	"""Dilate a 2D boolean mask with 8-connectivity using numpy only."""
	m = np.asarray(mask, dtype=bool)
	if m.ndim != 2:
		raise ValueError(f"mask must be 2D, got shape {m.shape}.")

	out = m.copy()
	for _ in range(max(0, int(iterations))):
		grown = out.copy()
		for dr in (-1, 0, 1):
			for dc in (-1, 0, 1):
				if dr == 0 and dc == 0:
					continue
				shift = np.roll(np.roll(out, dr, axis=0), dc, axis=1)
				if dr > 0:
					shift[:dr, :] = False
				elif dr < 0:
					shift[dr:, :] = False
				if dc > 0:
					shift[:, :dc] = False
				elif dc < 0:
					shift[:, dc:] = False
				grown |= shift
		out = grown

	return out


def detect_bad_pixel_clumps_from_ramp_fit(
	intercept_cube,
	slope_cube,
	scatter_cube=None,
	min_intercept_offset_dn=150.0,
	intercept_sigma=8.0,
	slope_percentile=2.0,
	scatter_sigma=8.0,
	dilate_iterations=1,
):
	"""Detect persistent bad/hot clumps using ramp-fit intercept/slope diagnostics."""
	intercept = np.asarray(intercept_cube, dtype=float)
	slope = np.asarray(slope_cube, dtype=float)
	if intercept.ndim != 3 or slope.ndim != 3:
		raise ValueError("intercept_cube and slope_cube must both be 3D arrays.")
	if intercept.shape != slope.shape:
		raise ValueError(
			f"intercept_cube shape {intercept.shape} must match slope_cube {slope.shape}."
		)

	pixel_intercept = np.nanmedian(intercept, axis=0)
	pixel_slope = np.nanmedian(slope, axis=0)

	i_med = float(np.nanmedian(pixel_intercept))
	i_mad = float(np.nanmedian(np.abs(pixel_intercept - i_med)))
	i_sig = 1.4826 * i_mad
	i_thr = max(i_med + intercept_sigma * i_sig, i_med + min_intercept_offset_dn)

	finite_slope = pixel_slope[np.isfinite(pixel_slope)]
	if finite_slope.size == 0:
		s_thr = 0.0
	else:
		s_thr = float(np.nanpercentile(finite_slope, slope_percentile))

	mask = (pixel_intercept > i_thr) & (pixel_slope < s_thr)

	if scatter_cube is not None:
		scatter = np.asarray(scatter_cube, dtype=float)
		if scatter.shape != intercept.shape:
			raise ValueError(
				f"scatter_cube shape {scatter.shape} must match intercept_cube {intercept.shape}."
			)
		pixel_scatter = np.nanmedian(scatter, axis=0)
		sc_med = float(np.nanmedian(pixel_scatter))
		sc_mad = float(np.nanmedian(np.abs(pixel_scatter - sc_med)))
		sc_sig = 1.4826 * sc_mad
		sc_thr = sc_med + scatter_sigma * sc_sig
		mask |= pixel_scatter > sc_thr
	else:
		sc_thr = np.nan

	mask |= ~np.isfinite(pixel_intercept) | ~np.isfinite(pixel_slope)

	for _ in range(max(0, int(dilate_iterations))):
		mask = dilate_binary_mask(mask, iterations=1)

	info = {
		"intercept_threshold_dn": float(i_thr),
		"slope_threshold": float(s_thr),
		"scatter_threshold": float(sc_thr) if np.isfinite(sc_thr) else None,
		"clump_fraction": float(np.mean(mask)),
		"n_clump_pixels": int(np.sum(mask)),
	}
	return mask, info


def correct_bad_pixels_with_neighbors(
	cube,
	bad_mask,
	max_radius=3,
	min_neighbors=4,
):
	"""Replace bad pixels/clumps with local robust neighbor medians per frame."""
	arr = np.asarray(cube, dtype=float)
	if arr.ndim != 3:
		raise ValueError(f"cube must be 3D, got shape {arr.shape}.")

	mask = np.asarray(bad_mask, dtype=bool)
	if mask.shape != arr.shape[1:]:
		raise ValueError(
			f"bad_mask shape {mask.shape} must match image plane {arr.shape[1:]}."
		)

	if not np.any(mask):
		return arr.copy(), {"n_fixed": 0, "fixed_fraction": 0.0}

	ny, nx = mask.shape
	idx = np.argwhere(mask)
	corrected = arr.copy()
	n_fixed = 0

	for f in range(corrected.shape[0]):
		frame = corrected[f]
		for r, c in idx:
			filled = False
			for rad in range(1, max(1, int(max_radius)) + 1):
				r0 = max(0, r - rad)
				r1 = min(ny, r + rad + 1)
				c0 = max(0, c - rad)
				c1 = min(nx, c + rad + 1)

				patch = frame[r0:r1, c0:c1]
				patch_mask = mask[r0:r1, c0:c1]
				vals = patch[~patch_mask]
				vals = vals[np.isfinite(vals)]

				if vals.size >= min_neighbors:
					frame[r, c] = float(np.median(vals))
					filled = True
					n_fixed += 1
					break

			if not filled:
				frame[r, c] = np.nan

		corrected[f] = frame

	info = {
		"n_fixed": int(n_fixed),
		"fixed_fraction": float(n_fixed / (mask.sum() * corrected.shape[0])),
	}
	return corrected, info


@njit(parallel=True, fastmath=True)
def _get_slope_cube_owls_numba(ramp_cube, times, read_noise=10.0, gain=1.0, threshold=4.0):
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
	intercept_cube = np.zeros((nint, nx, ny), dtype=np.float32)
	scatter_cube = np.zeros((nint, nx, ny), dtype=np.float32)

	dt = np.empty(ngroup - 1, dtype=np.float32)
	for g in range(ngroup - 1):
		dt[g] = times[g + 1] - times[g]

	for i in prange(nint):
		for x in range(nx):
			for y in range(ny):
				counts = np.empty(ngroup, dtype=np.float32)
				for g in range(ngroup):
					counts[g] = ramp_cube[i, g, x, y]

				rates = np.empty(ngroup - 1, dtype=np.float32)
				for g in range(ngroup - 1):
					rates[g] = (counts[g + 1] - counts[g]) / dt[g]

				med_rate = np.median(rates)

				abs_devs = np.empty(ngroup - 1, dtype=np.float32)
				for g in range(ngroup - 1):
					abs_devs[g] = np.abs(rates[g] - med_rate)
				mad = np.median(abs_devs)

				sigma = mad * 1.4826
				if sigma < 1e-6:
					sigma = np.std(rates) + 1e-6

				corrected_counts = counts.copy()
				for g in range(ngroup - 1):
					if np.abs(rates[g] - med_rate) > (threshold * sigma):
						expected_diff = med_rate * dt[g]
						actual_diff = counts[g + 1] - counts[g]
						excess = actual_diff - expected_diff

						for k in range(g + 1, ngroup):
							corrected_counts[k] -= excess

				sum_w = 0.0
				sum_wx = 0.0
				sum_wy = 0.0
				sum_wxx = 0.0
				sum_wxy = 0.0

				for g in range(ngroup):
					t = times[g]
					c = corrected_counts[g]
					variance = (read_noise ** 2) + (max(c, 0.0) / gain)
					w = 1.0 / variance

					sum_w += w
					sum_wx += w * t
					sum_wy += w * c
					sum_wxx += w * t * t
					sum_wxy += w * t * c

				delta = (sum_w * sum_wxx) - (sum_wx * sum_wx)

				if delta != 0.0:
					slope = (sum_w * sum_wxy - sum_wx * sum_wy) / delta
					intercept = (sum_wy - slope * sum_wx) / sum_w
					slope_cube[i, x, y] = slope
					intercept_cube[i, x, y] = intercept

					residual_sum = 0.0
					for g in range(ngroup):
						resid = corrected_counts[g] - (intercept + slope * times[g])
						residual_sum += resid * resid
					scatter_cube[i, x, y] = np.sqrt(residual_sum / ngroup)
				else:
					slope_cube[i, x, y] = 0.0
					intercept_cube[i, x, y] = 0.0
					scatter_cube[i, x, y] = 0.0

	return slope_cube, intercept_cube, scatter_cube


def get_slope_cube_owls(
	ramp_cube,
	times,
	read_noise=10.0,
	gain=1.0,
	threshold=4.0,
	return_diagnostics=False,
):
	"""Compute CR-rejected, optimally weighted least squares slopes.

	When return_diagnostics is True, also returns intercept and residual scatter cubes.
	"""
	slope_cube, intercept_cube, scatter_cube = _get_slope_cube_owls_numba(
		ramp_cube,
		times,
		read_noise=read_noise,
		gain=gain,
		threshold=threshold,
	)

	if return_diagnostics:
		return slope_cube, intercept_cube, scatter_cube

	return slope_cube


def _smooth_1d_boxcar(values, window):
    """Smooth a 1D array with a centered boxcar."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"values must be 1D, got shape {arr.shape}.")

    width = max(1, int(window))
    if width % 2 == 0:
        width += 1

    if width == 1:
        return arr.copy()

    kernel = np.ones(width, dtype=float) / float(width)
    padded = np.pad(arr, width // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def estimate_trace_aperture(
    cube,
    dispersion_min,
    dispersion_max,
    expected_spatial_center,
    max_half_width,
    smooth_window=5,
    threshold_sigma=2.5,
):
    """Estimate a spatial profile and simple peak locations for a trace."""
    arr = np.asarray(cube, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"cube must be 3D, got shape {arr.shape}.")

    n_frames, n_dispersion, n_spatial = arr.shape
    disp_lo = max(0, int(dispersion_min))
    disp_hi = min(n_dispersion - 1, int(dispersion_max))
    if disp_lo > disp_hi:
        raise ValueError(
            f"dispersion_min ({dispersion_min}) must be <= dispersion_max ({dispersion_max})."
        )

    dispersion_pixels = np.arange(disp_lo, disp_hi + 1, dtype=int)
    window = arr[:, disp_lo : disp_hi + 1, :]
    profile = np.nanmedian(window, axis=(0, 1))
    profile_smooth = _smooth_1d_boxcar(profile, smooth_window)

    search_lo = max(0, int(expected_spatial_center - max_half_width))
    search_hi = min(n_spatial - 1, int(expected_spatial_center + max_half_width))
    search_profile = profile_smooth[search_lo : search_hi + 1]

    if search_profile.size == 0:
        raise ValueError("Search window for aperture estimation is empty.")

    baseline = float(np.nanmedian(search_profile))
    deviation = float(np.nanmedian(np.abs(search_profile - baseline)))
    sigma = 1.4826 * deviation
    if not np.isfinite(sigma) or sigma == 0.0:
        sigma = float(np.nanstd(search_profile))
    if not np.isfinite(sigma) or sigma == 0.0:
        sigma = 1.0

    threshold = baseline + threshold_sigma * sigma

    peak_positions = []
    for idx in range(search_lo + 1, search_hi):
        value = profile_smooth[idx]
        if not np.isfinite(value):
            continue
        if value < threshold:
            continue
        if value >= profile_smooth[idx - 1] and value >= profile_smooth[idx + 1]:
            peak_positions.append(int(idx))

    if not peak_positions:
        peak_positions = [int(search_lo + int(np.nanargmax(search_profile)))]

    return {
        "profile": profile,
        "profile_smooth": profile_smooth,
        "threshold": float(threshold),
        "peak_positions": peak_positions,
        "spatial_axis": np.arange(n_spatial, dtype=int),
        "dispersion_pixels": dispersion_pixels,
        "search_bounds": (search_lo, search_hi),
        "expected_spatial_center": float(expected_spatial_center),
        "max_half_width": float(max_half_width),
    }


def build_linear_trace_aperture(
    n_dispersion,
    dispersion_min,
    dispersion_max,
    spatial_left_start,
    spatial_left_end,
    spatial_right_start,
    spatial_right_end,
    n_spatial,
):
    """Build a simple linearly varying photometric aperture."""
    n_dispersion = int(n_dispersion)
    n_spatial = int(n_spatial)
    if n_dispersion <= 0:
        raise ValueError("n_dispersion must be positive.")
    if n_spatial <= 0:
        raise ValueError("n_spatial must be positive.")

    dispersion_min = int(dispersion_min)
    dispersion_max = int(dispersion_max)
    if dispersion_min > dispersion_max:
        raise ValueError("dispersion_min must be <= dispersion_max.")

    dispersion_pixels = np.arange(dispersion_min, dispersion_max + 1, dtype=int)
    if dispersion_pixels.size != n_dispersion:
        raise ValueError(
            f"n_dispersion ({n_dispersion}) does not match dispersion range size ({dispersion_pixels.size})."
        )

    spatial_left = np.linspace(float(spatial_left_start), float(spatial_left_end), n_dispersion)
    spatial_right = np.linspace(float(spatial_right_start), float(spatial_right_end), n_dispersion)

    spatial_min = np.minimum(spatial_left, spatial_right)
    spatial_max = np.maximum(spatial_left, spatial_right)

    aperture_mask = np.zeros((n_dispersion, n_spatial), dtype=bool)
    for row_idx in range(n_dispersion):
        left = max(0, int(np.floor(spatial_min[row_idx])))
        right = min(n_spatial - 1, int(np.ceil(spatial_max[row_idx])))
        if left <= right:
            aperture_mask[row_idx, left : right + 1] = True

    return {
        "dispersion_pixels": dispersion_pixels,
        "spatial_left": spatial_left,
        "spatial_right": spatial_right,
        "spatial_left_int": np.floor(spatial_min).astype(int),
        "spatial_right_int": np.ceil(spatial_max).astype(int),
        "spatial_center": 0.5 * (spatial_left + spatial_right),
        "spatial_half_width": 0.5 * (spatial_right - spatial_left),
        "aperture_mask": aperture_mask,
    }


# ---------------------------------------------------------------------------
# Spectrophotometric extraction helpers
# ---------------------------------------------------------------------------

def extract_trace_spectra_variable_aperture(
    cube,
    aperture_model,
    bad_pixel_mask=None,
    subtract_background=False,
    background_inner_gap=2,
    background_width=8,
    background_mask=None,
    return_diagnostics=False,
):
    """Extract spectra using per-dispersion-pixel variable spatial bounds.

    When requested, estimate a local per-dispersion background from sidebands
    outside the aperture and subtract it after scaling by the aperture width.

    Extracts all good pixels equally (no spatial weighting). Wavelength-dependent
    weighting is applied separately when computing white-light photometry.

    Parameters
    ----------
    cube : 3D array, shape (n_int, n_dispersion, n_spatial)
    aperture_model : dict
        Output of ``build_linear_trace_aperture``.
    subtract_background : bool
        If True, estimate and subtract a local per-pixel background level.
    background_inner_gap : int
        Gap in pixels between aperture edge and background sideband.
    background_width : int
        Width of background sideband on each side.
    background_mask : 2D bool array (n_dispersion, n_spatial) or None
        Additional pixels to exclude from background estimation.
    return_diagnostics : bool
        If True return ``(spectra, dispersion, info_dict)``; else
        return ``(spectra, dispersion)``.

    Returns
    -------
    spectra : 2D array, shape (n_int, n_channels)
    dispersion : 1D int array, shape (n_channels,)
    info_dict (optional) : dict with keys
        ``background_per_pixel``, ``background_counts``,
        ``aperture_width_pixels``
    """
    arr = np.asarray(cube, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"cube must be 3D, got {arr.shape}.")

    disp = np.asarray(aperture_model["dispersion_pixels"], dtype=int)
    left = np.asarray(aperture_model["spatial_left_int"], dtype=int)
    right = np.asarray(aperture_model["spatial_right_int"], dtype=int)

    if not (disp.size == left.size == right.size):
        raise ValueError("Aperture model arrays must have matching lengths.")

    nint, ndisp, nspat = arr.shape
    spectra = np.full((nint, disp.size), np.nan, dtype=float)
    background_per_pixel = np.zeros((nint, disp.size), dtype=float)
    background_counts = np.zeros((nint, disp.size), dtype=int)
    aperture_width_pixels = np.zeros(disp.size, dtype=int)

    if bad_pixel_mask is None:
        bad = np.zeros((ndisp, nspat), dtype=bool)
    else:
        bad = np.asarray(bad_pixel_mask, dtype=bool)
        if bad.shape != (ndisp, nspat):
            raise ValueError(
                f"bad_pixel_mask must have shape {(ndisp, nspat)}, got {bad.shape}."
            )

    if background_mask is not None:
        bg_mask = np.asarray(background_mask, dtype=bool)
        if bg_mask.shape != (ndisp, nspat):
            raise ValueError(
                f"background_mask must have shape {(ndisp, nspat)}, got {bg_mask.shape}."
            )
    else:
        bg_mask = None

    for k, d in enumerate(disp):
        if d < 0 or d >= ndisp:
            raise ValueError(f"Dispersion pixel {d} out of bounds for cube shape {arr.shape}.")
        c0 = max(0, int(left[k]))
        c1 = min(nspat - 1, int(right[k]))
        if c1 < c0:
            continue
        aperture_width_pixels[k] = c1 - c0 + 1

        ap_vals = arr[:, d, c0 : c1 + 1]
        good_ap = np.isfinite(ap_vals) & (~bad[d, c0 : c1 + 1])[None, :]
        raw_sum = np.sum(np.where(good_ap, ap_vals, 0.0), axis=1)

        if subtract_background:
            bg_cols = np.zeros(nspat, dtype=bool)
            left_bg_lo = max(0, c0 - int(background_inner_gap) - int(background_width))
            left_bg_hi = c0 - int(background_inner_gap) - 1
            right_bg_lo = c1 + int(background_inner_gap) + 1
            right_bg_hi = min(nspat - 1, c1 + int(background_inner_gap) + int(background_width))
            if left_bg_hi >= left_bg_lo:
                bg_cols[left_bg_lo : left_bg_hi + 1] = True
            if right_bg_hi >= right_bg_lo:
                bg_cols[right_bg_lo : right_bg_hi + 1] = True
            if bg_mask is not None:
                bg_cols &= ~bg_mask[d]
            if np.any(bg_cols):
                bg_values = arr[:, d, bg_cols]
                bg_values = np.where(np.isfinite(bg_values), bg_values, np.nan)
                valid_counts = np.sum(np.isfinite(bg_values), axis=1)
                bg_level = np.nanmedian(bg_values, axis=1)
                bg_level = np.where(np.isfinite(bg_level), bg_level, 0.0)
                background_per_pixel[:, k] = bg_level
                background_counts[:, k] = valid_counts.astype(int)
                spectra[:, k] = raw_sum - bg_level * aperture_width_pixels[k]
            else:
                spectra[:, k] = raw_sum
        else:
            spectra[:, k] = raw_sum

    if return_diagnostics:
        return spectra, disp, {
            "background_per_pixel": background_per_pixel,
            "background_counts": background_counts,
            "aperture_width_pixels": aperture_width_pixels,
        }
    return spectra, disp


def build_channel_quality_mask(aperture_model, bad_pixel_mask):
    """Return a per-channel good-pixel flag from a 2D bad-pixel map.

    Parameters
    ----------
    aperture_model : dict
        Output of ``build_linear_trace_aperture``.
    bad_pixel_mask : 2D bool array, shape (n_dispersion, n_spatial)
        True where a detector pixel is bad / unreliable.

    Returns
    -------
    channel_good : 1D bool array, shape (n_channels,)
        True → channel is clean; False → contains at least one bad pixel.
    """
    mask = np.asarray(bad_pixel_mask, dtype=bool)
    disp = aperture_model["dispersion_pixels"]
    left = aperture_model["spatial_left_int"]
    right = aperture_model["spatial_right_int"]
    n_channels = len(disp)
    channel_good = np.ones(n_channels, dtype=bool)
    n_spatial = mask.shape[1]
    for k in range(n_channels):
        d = int(disp[k])
        c_lo = max(0, int(left[k]))
        c_hi = min(n_spatial - 1, int(right[k]))
        if np.any(mask[d, c_lo : c_hi + 1]):
            channel_good[k] = False
    return channel_good


def compute_aperture_motion_centroids(
    cube,
    aperture_model,
    background_per_pixel=None,
    bad_pixel_mask=None,
    guard_pixels=2,
):
    """Estimate spatial/dispersion centroids from an aperture trace footprint.

    Parameters
    ----------
    cube : 3D array, shape (n_int, n_dispersion, n_spatial)
    aperture_model : dict
        Output of ``build_linear_trace_aperture``.
    background_per_pixel : 2D array (n_int, n_channels) or None
        Background level per pixel per integration (from extraction diagnostics).
    bad_pixel_mask : 2D bool array (n_dispersion, n_spatial) or None
    guard_pixels : int
        Extra pixels beyond aperture edge included in centroid computation.

    Returns
    -------
    dict with keys ``spatial_centroid`` and ``dispersion_centroid``,
    each a 1D array of length n_int.
    """
    arr = np.asarray(cube, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"cube must be 3D, got {arr.shape}.")

    disp = np.asarray(aperture_model["dispersion_pixels"], dtype=int)
    left = np.asarray(aperture_model["spatial_left_int"], dtype=int)
    right = np.asarray(aperture_model["spatial_right_int"], dtype=int)
    if not (disp.size == left.size == right.size):
        raise ValueError("Aperture model arrays must have matching lengths.")

    nint, ndisp, nspat = arr.shape
    bg = (
        np.zeros((nint, disp.size), dtype=float)
        if background_per_pixel is None
        else np.asarray(background_per_pixel, dtype=float)
    )
    if bg.shape != (nint, disp.size):
        raise ValueError(
            f"background_per_pixel must have shape {(nint, disp.size)}, got {bg.shape}."
        )

    mask = (
        np.zeros((ndisp, nspat), dtype=bool)
        if bad_pixel_mask is None
        else np.asarray(bad_pixel_mask, dtype=bool)
    )

    spatial_centroid = np.full(nint, np.nan, dtype=float)
    dispersion_centroid = np.full(nint, np.nan, dtype=float)
    guard = int(guard_pixels)

    for i in range(nint):
        frame = arr[i]
        row_weights = np.zeros(disp.size, dtype=float)
        x_num, x_den = 0.0, 0.0
        for k, d in enumerate(disp):
            if d < 0 or d >= ndisp:
                continue
            l = max(0, int(left[k]) - guard)
            r = min(nspat - 1, int(right[k]) + guard)
            vals = frame[d, l : r + 1]
            good = np.isfinite(vals) & (~mask[d, l : r + 1])
            if not np.any(good):
                continue
            weights = np.where(good, vals - bg[i, k], 0.0)
            weights = np.clip(weights, 0.0, None)
            wsum = float(np.sum(weights))
            if wsum <= 0:
                continue
            x = np.arange(l, r + 1, dtype=float)
            x_num += float(np.sum(weights * x))
            x_den += wsum
            row_weights[k] = wsum
        if x_den > 0:
            spatial_centroid[i] = x_num / x_den
        if np.sum(row_weights) > 0:
            dispersion_centroid[i] = np.sum(row_weights * disp) / np.sum(row_weights)

    return {
        "spatial_centroid": spatial_centroid,
        "dispersion_centroid": dispersion_centroid,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _robust_mad_sigma(x):
    """Return (median, robust-sigma) for a 1-D array using 1.4826*MAD."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sig = 1.4826 * mad
    if (not np.isfinite(sig)) or sig <= 0:
        sig = np.nanstd(x)
    return med, max(float(sig), 1.0e-12)


def _pad_mask(mask, pad=1):
    """Extend False regions in a boolean mask by *pad* samples on each side."""
    mask = np.asarray(mask, dtype=bool).copy()
    if pad <= 0:
        return mask
    for idx in np.where(~mask)[0]:
        lo = max(0, idx - pad)
        hi = min(mask.size, idx + pad + 1)
        mask[lo:hi] = False
    return mask


# ---------------------------------------------------------------------------
# Photometric time-series helpers
# ---------------------------------------------------------------------------

def compute_wavelength_weighted_white_light(extracted_spectra, use_weights=True):
    """
    Compute white-light photometry with wavelength-dependent weighting.
    
    Brighter wavelengths (higher S/N) contribute more to the white-light curve.
    
    Parameters
    ----------
    extracted_spectra : 2D array, shape (n_int, n_channels)
        Extracted spectrum at each integration.
    use_weights : bool
        If True, weight by per-wavelength flux. If False, simple sum (unweighted).
        
    Returns
    -------
    white_light : 1D array, shape (n_int,)
        Weighted white-light photometry (in counts)
    wavelength_weights : 1D array, shape (n_channels,)
        Per-wavelength weights used
    """
    specs = np.asarray(extracted_spectra, dtype=float)
    if specs.ndim != 2:
        raise ValueError(f"extracted_spectra must be 2D, got shape {specs.shape}.")
    
    # Compute per-wavelength weights from median spectrum
    median_spectrum = np.nanmedian(specs, axis=0)
    
    if use_weights:
        # Normalize so median wavelength weight = 1
        weights = np.where(median_spectrum > 0, median_spectrum, 0.0)
        weights = weights / np.nanmedian(weights[weights > 0]) if np.any(weights > 0) else np.ones_like(weights)
    else:
        weights = np.ones(specs.shape[1], dtype=float)
    
    # Apply wavelength weights
    weighted_sum = np.nansum(specs * weights[None, :], axis=1)
    total_weight = np.nansum(weights)
    
    white_light = weighted_sum / total_weight if total_weight > 0 else weighted_sum
    
    return white_light, weights


def compute_white_light_products(
    extracted_spectra_masked,
    extraction_info,
    exposure_time_s,
    nint,
    ngroup,
    transit_ingress_index=0,
    transit_egress_index=None,
    use_wavelength_weights=True,
):
    """Compute white-light curve and per-channel statistics from masked spectra.
    
    Parameters
    ----------
    use_wavelength_weights : bool
        If True, weight white-light by per-wavelength flux (S/N weighting).
        If False, simple unweighted sum of all wavelengths.
    """
    if transit_egress_index is None:
        transit_egress_index = nint - 1

    if exposure_time_s is not None and exposure_time_s.size == nint * ngroup:
        trial = exposure_time_s.reshape(nint, ngroup)[:, -1]
        if np.nanmax(trial) > np.nanmin(trial):
            integration_time_axis = trial
            time_axis_label = "Ramp end time [s]"
        else:
            integration_time_axis = np.arange(nint, dtype=float)
            time_axis_label = "Integration index"
    else:
        integration_time_axis = np.arange(nint, dtype=float)
        time_axis_label = "Integration index"

    oot_mask = np.ones(nint, dtype=bool)
    oot_mask[int(transit_ingress_index) : int(transit_egress_index) + 1] = False

    # Compute wavelength-weighted white-light
    white_light, wavelength_weights = compute_wavelength_weighted_white_light(
        extracted_spectra_masked, use_weights=use_wavelength_weights
    )
    white_light_norm = white_light / np.nanmedian(white_light)
    median_spectrum = np.nanmedian(extracted_spectra_masked, axis=0)
    normalized_spectra = extracted_spectra_masked / median_spectrum[None, :]
    spectral_scatter_ppm = 1.0e6 * np.nanstd(normalized_spectra, axis=0)
    median_background_per_pixel = np.nanmedian(
        extraction_info["background_per_pixel"], axis=0
    )

    return {
        "integration_time_axis": integration_time_axis,
        "time_axis_label": time_axis_label,
        "oot_mask": oot_mask,
        "white_light": white_light,
        "white_light_norm": white_light_norm,
        "median_spectrum": median_spectrum,
        "normalized_spectra": normalized_spectra,
        "spectral_scatter_ppm": spectral_scatter_ppm,
        "median_background_per_pixel": median_background_per_pixel,
    }


def reject_photometric_excursions(
    white_light_norm,
    dx,
    dy,
    oot_mask,
    flux_window_low=0.97,
    flux_window_high=1.03,
    excursion_sigma=6.0,
    excursion_padding=1,
    motion_sigma=6.0,
):
    """Flag integrations with flux excursions or large pointing jumps."""
    wl = np.asarray(white_light_norm, dtype=float)
    n = wl.size

    diff_wl = np.diff(wl, prepend=wl[0])
    trend = np.array(
        [float(np.nanmedian(wl[max(0, i - 4) : min(n, i + 5)])) for i in range(n)]
    )
    resid_wl = wl - trend

    wl_med, wl_sig = _robust_mad_sigma(resid_wl[oot_mask])
    dw_med, dw_sig = _robust_mad_sigma(diff_wl[oot_mask])

    motion_delta = np.sqrt(np.diff(dx, prepend=dx[0]) ** 2 + np.diff(dy, prepend=dy[0]) ** 2)
    motion_med, motion_sig_val = _robust_mad_sigma(motion_delta[oot_mask])

    keep = np.isfinite(wl)
    keep &= wl >= float(flux_window_low)
    keep &= wl <= float(flux_window_high)
    keep &= np.abs(resid_wl - wl_med) <= float(excursion_sigma) * wl_sig
    keep &= np.abs(diff_wl - dw_med) <= float(excursion_sigma) * dw_sig
    keep &= motion_delta <= motion_med + float(motion_sigma) * motion_sig_val
    keep = _pad_mask(keep, pad=int(excursion_padding))
    return keep


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_spatial_profile(trace_est, trace_params):
    """Plot the collapsed spatial profile used to establish the trace aperture."""
    spat_x = np.arange(trace_est["profile_smooth"].size)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(spat_x, trace_est["profile"], alpha=0.5, label="Raw profile")
    ax.plot(spat_x, trace_est["profile_smooth"], lw=2, label="Smoothed profile")
    ax.axvline(trace_params["spatial_left_start"], color="r", ls="--", label="Start bounds")
    ax.axvline(trace_params["spatial_right_start"], color="r", ls="--")
    ax.axvline(trace_params["spatial_left_end"], color="orange", ls=":", label="End bounds")
    ax.axvline(trace_params["spatial_right_end"], color="orange", ls=":")
    ax.axhline(trace_est["threshold"], color="k", ls=":", label="Threshold")
    for peak in trace_est["peak_positions"]:
        ax.axvline(peak, color="g", ls="-.", alpha=0.8)
    ax.set_xlabel("Spatial pixel")
    ax.set_ylabel("Collapsed slope signal")
    ax.set_title("Global Spatial Profile and Variable Aperture Priors")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_aperture_overlay(slope_cube, aperture_model, trace_est, image_index=20):
    """Overlay the variable aperture bounds on a single slope integration."""
    image_index = min(image_index, slope_cube.shape[0] - 1)
    img = slope_cube[image_index]
    finite = np.isfinite(img)
    vmin = float(np.nanpercentile(img[finite], 5)) if np.any(finite) else 0.0
    vmax = float(np.nanpercentile(img[finite], 99.5)) if np.any(finite) else 1.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.plot(
        aperture_model["spatial_left"],
        aperture_model["dispersion_pixels"],
        color="r",
        lw=1.8,
        ls="--",
        label="Variable aperture",
    )
    ax.plot(
        aperture_model["spatial_right"],
        aperture_model["dispersion_pixels"],
        color="r",
        lw=1.8,
        ls="--",
    )
    for peak in trace_est["peak_positions"]:
        ax.axvline(peak, color="cyan", ls="-.", lw=1.2, alpha=0.8, label="Detected peak")
    ax.set_xlim(0, img.shape[1] - 1)
    ax.set_ylim(0, img.shape[0] - 1)
    ax.set_xlabel("Spatial pixel")
    ax.set_ylabel("Dispersion pixel")
    ax.set_title(f"Trace/Aperture Overlay on Corrected Slope Image (integration {image_index})")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_spectrophotometry_diagnostics(
    extracted_dispersion,
    median_spectrum,
    channel_good,
    integration_time_axis,
    white_light_norm,
    white_light_clean,
    median_background_per_pixel,
    normalized_spectra,
    dx,
    oot_mask,
    spectral_scatter_ppm,
    time_axis_label,
):
    """Generate a six-panel spectrophotometry diagnostic figure."""
    n_bad_channels = int(np.sum(~channel_good))
    time_lo = float(np.nanmin(integration_time_axis))
    time_hi = float(np.nanmax(integration_time_axis))
    if time_hi <= time_lo:
        time_hi = time_lo + 1.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(extracted_dispersion, median_spectrum, color="tab:blue", lw=1.5)
    for bd in extracted_dispersion[~channel_good]:
        ax.axvline(bd, color="red", lw=0.8, alpha=0.5)
    ax.set_xlabel("Dispersion pixel")
    ax.set_ylabel("Median aperture flux")
    ax.set_title(f"Median Extracted Spectrum ({n_bad_channels} bad ch. marked red)")

    ax = axes[0, 1]
    ax.plot(integration_time_axis, white_light_norm, marker=".", lw=1, color="0.55", label="All integrations")
    ax.plot(integration_time_axis, white_light_clean, lw=1.2, color="tab:green", label="Kept integrations")
    in_transit = (~oot_mask) & np.isfinite(white_light_norm)
    if np.any(in_transit):
        t_in = integration_time_axis[in_transit]
        f_in = white_light_norm[in_transit]
        ax.scatter(
            t_in,
            f_in,
            s=16,
            color="tab:orange",
            edgecolor="none",
            alpha=0.9,
            label="In-transit (raw)",
            zorder=4,
        )
        ax.axvspan(np.nanmin(t_in), np.nanmax(t_in), color="tab:orange", alpha=0.12)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel(time_axis_label)
    ax.set_ylabel("Normalized white-light flux")
    ax.set_title("White-light Curve")
    ax.set_ylim(0.97, 1.03)
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 2]
    ax.plot(extracted_dispersion, median_background_per_pixel, color="tab:orange", lw=1.5)
    ax.set_xlabel("Dispersion pixel")
    ax.set_ylabel("Background per pixel")
    ax.set_title("Local Background Spectrum")

    ax = axes[1, 0]
    im = ax.imshow(
        normalized_spectra,
        origin="lower",
        aspect="auto",
        extent=[extracted_dispersion[0], extracted_dispersion[-1], time_lo, time_hi],
        vmin=0.97,
        vmax=1.03,
        cmap="magma",
    )
    ax.set_xlabel("Dispersion pixel")
    ax.set_ylabel(time_axis_label)
    ax.set_title("Normalized Spectral Time Series")
    fig.colorbar(im, ax=ax, label="Relative flux")

    ax = axes[1, 1]
    ax.scatter(
        dx,
        white_light_norm - np.nanmedian(white_light_norm[oot_mask]),
        s=12,
        alpha=0.6,
        color="tab:purple",
    )
    ax.axhline(0.0, color="k", ls=":", lw=1)
    ax.set_xlabel("dx [pix]")
    ax.set_ylabel("White-light residual")
    ax.set_title("Pointing Correlation Diagnostic")

    ax = axes[1, 2]
    ax.plot(extracted_dispersion, spectral_scatter_ppm, color="0.55", lw=1.0)
    for bd in extracted_dispersion[~channel_good]:
        ax.axvline(bd, color="red", lw=0.8, alpha=0.5, ls="--")
    ax.set_xlabel("Dispersion pixel")
    ax.set_ylabel("Scatter [ppm]")
    ax.set_title("Per-channel Temporal Scatter")

    plt.show()


def plot_flux_motion_correlation(dx, dy, white_light_norm, integration_time_axis, time_axis_label):
    """Scatter plot of flux versus pointing motion radius."""
    motion_radius = np.maximum(np.sqrt(dx * dx + dy * dy), 1.0e-6)
    finite = np.isfinite(motion_radius) & np.isfinite(white_light_norm)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    sc = ax.scatter(
        motion_radius[finite],
        white_light_norm[finite],
        s=14,
        alpha=0.7,
        c=integration_time_axis[finite],
        cmap="viridis",
    )
    fig.colorbar(sc, ax=ax, label=time_axis_label)
    ax.set_xscale("log")
    ax.set_xlabel("Motion radius sqrt(dx^2 + dy^2) [pix] (log)")
    ax.set_ylabel("Normalized white-light flux")
    ax.set_ylim(0.97, 1.03)
    ax.set_title("Flux-Motion Correlation")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    plt.show()


def _convolve1d_same_axis(image_2d, kernel_1d, axis):
    """Convolve a 2D image with a 1D kernel along one axis using edge padding."""
    arr = np.asarray(image_2d, dtype=float)
    ker = np.asarray(kernel_1d, dtype=float).reshape(-1)
    if arr.ndim != 2:
        raise ValueError(f"image_2d must be 2D, got {arr.shape}.")
    if ker.size < 3 or (ker.size % 2) != 1:
        raise ValueError("kernel_1d must have odd length >= 3.")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")

    pad = ker.size // 2
    out = np.zeros_like(arr, dtype=float)
    if axis == 0:
        padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
        for j, coeff in enumerate(ker):
            out += coeff * padded[j : j + arr.shape[0], :]
    else:
        padded = np.pad(arr, ((0, 0), (pad, pad)), mode="edge")
        for j, coeff in enumerate(ker):
            out += coeff * padded[:, j : j + arr.shape[1]]
    return out


def _shift_image_subpixel(image_2d, shift_x=0.0, shift_y=0.0, fill_value=np.nan):
    """Shift image by (shift_x, shift_y) using bilinear interpolation.

    Positive shift_x moves content to larger column index (right).
    Positive shift_y moves content to larger row index (down in array index).
    """
    img = np.asarray(image_2d, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"image_2d must be 2D, got {img.shape}.")

    ny, nx = img.shape
    yy, xx = np.indices((ny, nx), dtype=float)
    xs = xx - float(shift_x)
    ys = yy - float(shift_y)

    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = xs - x0
    wy = ys - y0

    def sample(yi, xi):
        out = np.full((ny, nx), fill_value, dtype=float)
        inside = (yi >= 0) & (yi < ny) & (xi >= 0) & (xi < nx)
        if np.any(inside):
            out[inside] = img[yi[inside], xi[inside]]
        return out

    v00 = sample(y0, x0)
    v01 = sample(y0, x1)
    v10 = sample(y1, x0)
    v11 = sample(y1, x1)

    w00 = (1.0 - wx) * (1.0 - wy)
    w01 = wx * (1.0 - wy)
    w10 = (1.0 - wx) * wy
    w11 = wx * wy

    out = np.full((ny, nx), fill_value, dtype=float)
    finite = np.isfinite(v00) | np.isfinite(v01) | np.isfinite(v10) | np.isfinite(v11)
    if np.any(finite):
        num = np.zeros((ny, nx), dtype=float)
        den = np.zeros((ny, nx), dtype=float)
        for vv, ww in ((v00, w00), (v01, w01), (v10, w10), (v11, w11)):
            ok = np.isfinite(vv)
            num[ok] += vv[ok] * ww[ok]
            den[ok] += ww[ok]
        valid = den > 0
        out[valid] = num[valid] / den[valid]
    return out


def build_aligned_master_frame(
    cube,
    shift_x,
    shift_y,
    use_mask=None,
):
    """Build a master frame from a cube after subpixel alignment using shifts."""
    arr = np.asarray(cube, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"cube must be 3D, got {arr.shape}.")
    nint = arr.shape[0]
    sx = np.asarray(shift_x, dtype=float).reshape(-1)
    sy = np.asarray(shift_y, dtype=float).reshape(-1)
    if sx.shape != (nint,) or sy.shape != (nint,):
        raise ValueError("shift_x and shift_y must match integration axis length.")

    if use_mask is None:
        use = np.ones(nint, dtype=bool)
    else:
        use = np.asarray(use_mask, dtype=bool)
        if use.shape != (nint,):
            raise ValueError("use_mask must match integration axis length.")

    aligned = []
    for i in range(nint):
        if not use[i] or (not np.isfinite(sx[i])) or (not np.isfinite(sy[i])):
            continue
        aligned.append(_shift_image_subpixel(arr[i], shift_x=-sx[i], shift_y=-sy[i]))
    if len(aligned) == 0:
        raise ValueError("No valid frames available to build aligned master frame.")
    return np.nanmedian(np.asarray(aligned), axis=0)


def estimate_difference_image_shifts(
    cube,
    reference_image=None,
    bad_pixel_mask=None,
    kernel_dx=(-0.5, 0.0, 0.5),
    kernel_dy=(-0.5, 0.0, 0.5),
    background_order=1,
    clip_sigma=8.0,
    min_valid_pixels=1000,
):
    """Estimate per-integration dx/dy from difference images.

    The model solves, per integration,
    ``frame - reference ~= a*grad_x + b*grad_y + background_terms``
    then reports ``dx=-a`` and ``dy=-b``.  Background terms include a constant
    and, when ``background_order>=1``, a linear plane (x and y terms).
    """
    arr = np.asarray(cube, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"cube must be 3D, got {arr.shape}.")
    nint, nrow, ncol = arr.shape

    if reference_image is None:
        ref = np.nanmedian(arr, axis=0)
    else:
        ref = np.asarray(reference_image, dtype=float)
        if ref.shape != (nrow, ncol):
            raise ValueError(f"reference_image must have shape {(nrow, ncol)}, got {ref.shape}.")

    if bad_pixel_mask is None:
        bad = np.zeros((nrow, ncol), dtype=bool)
    else:
        bad = np.asarray(bad_pixel_mask, dtype=bool)
        if bad.shape != (nrow, ncol):
            raise ValueError(f"bad_pixel_mask must have shape {(nrow, ncol)}, got {bad.shape}.")

    gx = _convolve1d_same_axis(ref, kernel_dx, axis=1)
    gy = _convolve1d_same_axis(ref, kernel_dy, axis=0)

    yy, xx = np.indices((nrow, ncol), dtype=float)
    xnorm = (xx - 0.5 * (ncol - 1)) / max(1.0, 0.5 * (ncol - 1))
    ynorm = (yy - 0.5 * (nrow - 1)) / max(1.0, 0.5 * (nrow - 1))

    dx = np.full(nint, np.nan, dtype=float)
    dy = np.full(nint, np.nan, dtype=float)
    bg0 = np.full(nint, np.nan, dtype=float)
    bgx = np.full(nint, np.nan, dtype=float)
    bgy = np.full(nint, np.nan, dtype=float)
    residual_rms = np.full(nint, np.nan, dtype=float)
    n_valid = np.zeros(nint, dtype=int)

    base_valid = np.isfinite(ref) & np.isfinite(gx) & np.isfinite(gy) & (~bad)

    for i in range(nint):
        diff = arr[i] - ref
        valid = base_valid & np.isfinite(diff)

        if clip_sigma is not None and np.any(valid):
            med = np.nanmedian(diff[valid])
            mad = np.nanmedian(np.abs(diff[valid] - med))
            sig = 1.4826 * mad
            if np.isfinite(sig) and sig > 0:
                valid &= np.abs(diff - med) <= float(clip_sigma) * sig

        n_valid[i] = int(np.sum(valid))
        if n_valid[i] < max(int(min_valid_pixels), 8):
            continue

        cols = [gx[valid], gy[valid], np.ones(n_valid[i], dtype=float)]
        if int(background_order) >= 1:
            cols.append(xnorm[valid])
            cols.append(ynorm[valid])
        A = np.column_stack(cols)
        b = diff[valid]

        if A.shape[0] <= A.shape[1]:
            continue

        beta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        model = A @ beta
        resid = b - model

        dx[i] = -float(beta[0])
        dy[i] = -float(beta[1])
        bg0[i] = float(beta[2])
        if int(background_order) >= 1:
            bgx[i] = float(beta[3])
            bgy[i] = float(beta[4])
        residual_rms[i] = float(np.nanstd(resid))

    return {
        "dx": dx,
        "dy": dy,
        "background_offset": bg0,
        "background_x": bgx,
        "background_y": bgy,
        "residual_rms": residual_rms,
        "n_valid_pixels": n_valid,
        "reference_image": ref,
        "gradient_x": gx,
        "gradient_y": gy,
    }


def plot_difference_imaging_motion(
    integration_time_axis,
    dx_diff,
    dy_diff,
    oot_mask=None,
    dx_reference=None,
    dy_reference=None,
    reference_label="Aperture-centroid motion",
):
    """Plot difference-imaging dx/dy with optional comparison motion series."""
    t = np.asarray(integration_time_axis, dtype=float)
    dx_arr = np.asarray(dx_diff, dtype=float)
    dy_arr = np.asarray(dy_diff, dtype=float)
    if t.shape != dx_arr.shape or t.shape != dy_arr.shape:
        raise ValueError("integration_time_axis, dx_diff, and dy_diff must have matching shapes.")

    if oot_mask is None:
        oot = np.ones_like(dx_arr, dtype=bool)
    else:
        oot = np.asarray(oot_mask, dtype=bool)
        if oot.shape != dx_arr.shape:
            raise ValueError("oot_mask must match the shape of dx_diff.")
    in_tr = ~oot

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(t, dx_arr, color="tab:blue", lw=1.2, label="Difference imaging")
    if np.any(in_tr):
        ax.scatter(t[in_tr], dx_arr[in_tr], s=14, color="tab:orange", alpha=0.85, label="In-transit")
    if dx_reference is not None:
        dx_ref = np.asarray(dx_reference, dtype=float)
        if dx_ref.shape == dx_arr.shape:
            ax.plot(t, dx_ref, color="0.45", lw=1.0, ls="--", label=reference_label)
    ax.axhline(0.0, color="k", lw=1, ls=":")
    ax.set_ylabel("dx [pix]")
    ax.set_title("Difference-Image Motion: dx")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.plot(t, dy_arr, color="tab:green", lw=1.2, label="Difference imaging")
    if np.any(in_tr):
        ax.scatter(t[in_tr], dy_arr[in_tr], s=14, color="tab:orange", alpha=0.85, label="In-transit")
    if dy_reference is not None:
        dy_ref = np.asarray(dy_reference, dtype=float)
        if dy_ref.shape == dy_arr.shape:
            ax.plot(t, dy_ref, color="0.45", lw=1.0, ls="--", label=reference_label)
    ax.axhline(0.0, color="k", lw=1, ls=":")
    ax.set_xlabel("Time / integration")
    ax.set_ylabel("dy [pix]")
    ax.set_title("Difference-Image Motion: dy")
    ax.legend(loc="best", fontsize=8)

    plt.show()


def plot_white_light_vs_dx_log(
    dx_motion,
    white_light_norm,
    oot_mask=None,
    integration_keep_mask=None,
    dx_label="dx",
):
    """Plot normalized white-light photometry against |dx| using log x-scale."""
    dx_arr = np.asarray(dx_motion, dtype=float)
    wl_arr = np.asarray(white_light_norm, dtype=float)
    if dx_arr.shape != wl_arr.shape:
        raise ValueError("dx_motion and white_light_norm must have matching shapes.")

    finite = np.isfinite(dx_arr) & np.isfinite(wl_arr)
    x_abs = np.maximum(np.abs(dx_arr), 1.0e-6)

    if oot_mask is None:
        oot = np.ones_like(dx_arr, dtype=bool)
    else:
        oot = np.asarray(oot_mask, dtype=bool)
        if oot.shape != dx_arr.shape:
            raise ValueError("oot_mask must match dx_motion shape.")
    in_transit = (~oot) & finite
    oot_rows = oot & finite

    if integration_keep_mask is None:
        keep = np.ones_like(dx_arr, dtype=bool)
    else:
        keep = np.asarray(integration_keep_mask, dtype=bool)
        if keep.shape != dx_arr.shape:
            raise ValueError("integration_keep_mask must match dx_motion shape.")
    rejected = (~keep) & finite

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if np.any(oot_rows):
        ax.scatter(
            x_abs[oot_rows],
            wl_arr[oot_rows],
            s=14,
            alpha=0.7,
            color="tab:blue",
            label="Out-of-transit",
        )
    if np.any(in_transit):
        ax.scatter(
            x_abs[in_transit],
            wl_arr[in_transit],
            s=16,
            alpha=0.85,
            color="tab:orange",
            label="In-transit",
        )
    if np.any(rejected):
        ax.scatter(
            x_abs[rejected],
            wl_arr[rejected],
            s=14,
            alpha=0.45,
            color="0.55",
            label="Rejected",
        )

    ax.set_xscale("log")
    ax.set_ylim(0.97, 1.03)
    ax.set_xlabel(f"|{dx_label}| [pix] (log)")
    ax.set_ylabel("Normalized white-light flux")
    ax.set_title("White-light Photometry vs dx Motion")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="best", fontsize=8)
    plt.show()


def extract_difference_image_photometry(
    cube,
    aperture_model,
    reference_image=None,
    shift_x=None,
    shift_y=None,
    bad_pixel_mask=None,
    subtract_background=True,
    background_inner_gap=2,
    background_width=10,
    background_mask=None,
    use_aperture_weights=True,
    weight_sigma_scale=0.45,
    return_diagnostics=False,
):
    """Extract aperture photometry from difference images.

    For each integration, this computes aperture flux as:
    reference-aperture flux + difference-image aperture delta,
    with optional local sideband background subtraction in the difference image.
    """
    arr = np.asarray(cube, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"cube must be 3D, got {arr.shape}.")
    nint, ndisp, nspat = arr.shape

    disp = np.asarray(aperture_model["dispersion_pixels"], dtype=int)
    left = np.asarray(aperture_model["spatial_left_int"], dtype=int)
    right = np.asarray(aperture_model["spatial_right_int"], dtype=int)
    if not (disp.size == left.size == right.size):
        raise ValueError("Aperture model arrays must have matching lengths.")

    if reference_image is None:
        ref = np.nanmedian(arr, axis=0)
    else:
        ref = np.asarray(reference_image, dtype=float)
        if ref.shape != (ndisp, nspat):
            raise ValueError(f"reference_image must have shape {(ndisp, nspat)}, got {ref.shape}.")

    if shift_x is None:
        sx = np.zeros(nint, dtype=float)
    else:
        sx = np.asarray(shift_x, dtype=float).reshape(-1)
        if sx.shape != (nint,):
            raise ValueError("shift_x must have shape (n_int,).")
    if shift_y is None:
        sy = np.zeros(nint, dtype=float)
    else:
        sy = np.asarray(shift_y, dtype=float).reshape(-1)
        if sy.shape != (nint,):
            raise ValueError("shift_y must have shape (n_int,).")

    if bad_pixel_mask is None:
        bad = np.zeros((ndisp, nspat), dtype=bool)
    else:
        bad = np.asarray(bad_pixel_mask, dtype=bool)
        if bad.shape != (ndisp, nspat):
            raise ValueError(f"bad_pixel_mask must have shape {(ndisp, nspat)}, got {bad.shape}.")

    if background_mask is None:
        bg_exclude = bad.copy()
    else:
        bg_exclude = np.asarray(background_mask, dtype=bool)
        if bg_exclude.shape != (ndisp, nspat):
            raise ValueError(
                f"background_mask must have shape {(ndisp, nspat)}, got {bg_exclude.shape}."
            )
        bg_exclude |= bad

    spectra_total = np.full((nint, disp.size), np.nan, dtype=float)
    spectra_delta = np.full((nint, disp.size), np.nan, dtype=float)
    reference_spectrum = np.full(disp.size, np.nan, dtype=float)
    background_per_pixel = np.zeros((nint, disp.size), dtype=float)
    background_counts = np.zeros((nint, disp.size), dtype=int)
    aperture_width_pixels = np.zeros((nint, disp.size), dtype=int)

    for k, d in enumerate(disp):
        if d < 0 or d >= ndisp:
            raise ValueError(f"Dispersion pixel {d} is out of bounds for cube shape {arr.shape}.")

        c0 = max(0, int(left[k]))
        c1 = min(nspat - 1, int(right[k]))
        if c1 < c0:
            continue

        ref_row = ref[d, c0 : c1 + 1]
        bad_row = bad[d, c0 : c1 + 1]
        ref_good = np.isfinite(ref_row) & (~bad_row)
        if not np.any(ref_good):
            continue

        cols = np.arange(c0, c1 + 1, dtype=float)
        if use_aperture_weights:
            center = 0.5 * (float(left[k]) + float(right[k]))
            sigma = max(1.0, float(weight_sigma_scale) * max(1.0, float(c1 - c0 + 1)))
            w = np.exp(-0.5 * ((cols - center) / sigma) ** 2)
            w = np.where(np.isfinite(w), w, 0.0)
            if np.sum(w) > 0:
                w = w / np.mean(w)
        else:
            w = np.ones_like(cols, dtype=float)

        ref_flux = float(np.sum(ref_row[ref_good] * w[ref_good]))
        delta_ap = np.full(nint, np.nan, dtype=float)
        widths = np.zeros(nint, dtype=int)

        for i in range(nint):
            frame_i = _shift_image_subpixel(arr[i], shift_x=-sx[i], shift_y=-sy[i])
            row_i = frame_i[d, c0 : c1 + 1]
            good_i = np.isfinite(row_i) & ref_good
            widths[i] = int(np.sum(good_i))
            if widths[i] == 0:
                delta_ap[i] = np.nan
                continue
            delta_ap[i] = float(np.sum((row_i[good_i] - ref_row[good_i]) * w[good_i]))

        aperture_width_pixels[:, k] = widths

        if subtract_background:
            bg_cols = np.zeros(nspat, dtype=bool)
            left_bg_lo = max(0, c0 - int(background_inner_gap) - int(background_width))
            left_bg_hi = c0 - int(background_inner_gap) - 1
            right_bg_lo = c1 + int(background_inner_gap) + 1
            right_bg_hi = min(nspat - 1, c1 + int(background_inner_gap) + int(background_width))
            if left_bg_hi >= left_bg_lo:
                bg_cols[left_bg_lo : left_bg_hi + 1] = True
            if right_bg_hi >= right_bg_lo:
                bg_cols[right_bg_lo : right_bg_hi + 1] = True
            bg_cols &= ~bg_exclude[d]

            if np.any(bg_cols):
                ref_bg = ref[d, bg_cols]
                diff_bg = np.full((nint, int(np.sum(bg_cols))), np.nan, dtype=float)
                for i in range(nint):
                    frame_i = _shift_image_subpixel(arr[i], shift_x=-sx[i], shift_y=-sy[i])
                    diff_bg[i] = frame_i[d, bg_cols] - ref_bg
                bg_level = np.nanmedian(diff_bg, axis=1)
                bg_level = np.where(np.isfinite(bg_level), bg_level, 0.0)
                background_per_pixel[:, k] = bg_level
                background_counts[:, k] = np.sum(np.isfinite(diff_bg), axis=1).astype(int)

                ref_bg_vals = ref_bg
                ref_bg = float(np.nanmedian(ref_bg_vals[np.isfinite(ref_bg_vals)])) if np.any(np.isfinite(ref_bg_vals)) else 0.0
                ref_flux -= ref_bg * float(np.sum(ref_good * w))

                delta_ap = delta_ap - bg_level * widths.astype(float)

        reference_spectrum[k] = ref_flux
        spectra_delta[:, k] = delta_ap
        spectra_total[:, k] = ref_flux + delta_ap

    if return_diagnostics:
        return spectra_total, disp, {
            "spectra_delta": spectra_delta,
            "reference_spectrum": reference_spectrum,
            "background_per_pixel": background_per_pixel,
            "background_counts": background_counts,
            "aperture_width_pixels": aperture_width_pixels,
            "reference_image": ref,
        }
    return spectra_total, disp


def plot_difference_image_photometry_comparison(
    integration_time_axis,
    white_light_norm,
    white_light_diff_norm,
    oot_mask=None,
    integration_keep_mask=None,
    label_base="Aperture photometry",
    label_diff="Difference-image photometry",
):
    """Compare standard and difference-image white-light photometry."""
    t = np.asarray(integration_time_axis, dtype=float)
    w0 = np.asarray(white_light_norm, dtype=float)
    wd = np.asarray(white_light_diff_norm, dtype=float)
    if t.shape != w0.shape or t.shape != wd.shape:
        raise ValueError("integration_time_axis, white_light_norm, and white_light_diff_norm must match.")

    if oot_mask is None:
        oot = np.ones_like(w0, dtype=bool)
    else:
        oot = np.asarray(oot_mask, dtype=bool)
    if integration_keep_mask is None:
        keep = np.ones_like(w0, dtype=bool)
    else:
        keep = np.asarray(integration_keep_mask, dtype=bool)

    finite = np.isfinite(w0) & np.isfinite(wd)
    resid = wd - w0
    in_tr = (~oot) & finite
    rej = (~keep) & finite

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(t, w0, lw=1.1, color="0.5", label=label_base)
    ax.plot(t, wd, lw=1.2, color="tab:blue", label=label_diff)
    if np.any(in_tr):
        ax.scatter(t[in_tr], wd[in_tr], s=14, color="tab:orange", alpha=0.85, label="In-transit")
    if np.any(rej):
        ax.scatter(t[rej], wd[rej], s=12, color="0.35", alpha=0.5, label="Rejected")
    ax.axhline(1.0, color="k", lw=1, ls=":")
    ax.set_ylabel("Normalized white-light flux")
    ax.set_title("Difference-Image vs Standard White-Light Photometry")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.plot(t, resid, lw=1.0, color="tab:purple")
    ax.axhline(0.0, color="k", lw=1, ls=":")
    ax.set_xlabel("Time / integration")
    ax.set_ylabel("Diff - standard")
    ax.set_title("Photometry Difference Residual")
    ax.grid(True, ls=":", alpha=0.4)

    plt.show()


def list_fits_telemetry_channels(fits_path):
    """List available FITS table telemetry channels by HDU and column name."""
    out = {}
    with fits.open(fits_path, memmap=True) as hdul:
        for hdu in hdul:
            cols = getattr(hdu.data, "columns", None)
            if cols is None:
                continue
            names = [c.name for c in cols]
            if names:
                out[hdu.name] = names
    return out


def extract_numeric_header_cards(header):
    """Return numeric FITS header cards as a dictionary."""
    numeric = {}
    for key, value in header.items():
        if key in {"", "COMMENT", "HISTORY"}:
            continue
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            numeric[str(key)] = float(value)
    return numeric


def sample_fits_telemetry_to_integrations(
    fits_path,
    integration_jd,
    hdu_names=("VITL_DATA", "SC_QUATERNIONS", "SC_POSITION", "SC_VELOCITY"),
    exclude_columns=("jd",),
):
    """Interpolate FITS table telemetry columns onto integration JD timestamps."""
    t = np.asarray(integration_jd, dtype=float).reshape(-1)
    telemetry = {}
    with fits.open(fits_path, memmap=True) as hdul:
        for hname in hdu_names:
            if hname not in hdul:
                continue
            tab = hdul[hname].data
            cols = getattr(tab, "columns", None)
            if cols is None or "jd" not in [c.name for c in cols]:
                continue

            jd = np.asarray(tab["jd"], dtype=float).reshape(-1)
            good_jd = np.isfinite(jd)
            if np.sum(good_jd) < 4:
                continue
            jd = jd[good_jd]

            sort_idx = np.argsort(jd)
            jd = jd[sort_idx]

            for col in cols.names:
                if col.lower() in {str(x).lower() for x in exclude_columns}:
                    continue
                y_raw = np.asarray(tab[col], dtype=float).reshape(-1)
                y_raw = y_raw[good_jd][sort_idx]
                good = np.isfinite(y_raw)
                if np.sum(good) < 4:
                    continue

                # np.interp does nearest-edge extrapolation at boundaries.
                y_interp = np.interp(t, jd[good], y_raw[good])
                telemetry[f"{hname}.{col}"] = y_interp
    return telemetry


def compute_multimetric_correlation_and_pca(
    metric_dict,
    white_key="white_light_norm",
    oot_mask=None,
    n_components=4,
):
    """Build correlation matrix and PCA decomposition for 1D time-series metrics."""
    keys = [k for k, v in metric_dict.items() if np.asarray(v).ndim == 1]
    if len(keys) < 2:
        raise ValueError("Need at least two 1D metrics for correlation/PCA diagnostics.")

    lengths = [np.asarray(metric_dict[k]).size for k in keys]
    n = lengths[0]
    if any(m != n for m in lengths):
        raise ValueError("All metric arrays must have the same length.")

    X = np.column_stack([np.asarray(metric_dict[k], dtype=float) for k in keys])
    finite_rows = np.all(np.isfinite(X), axis=1)
    if oot_mask is not None:
        oot = np.asarray(oot_mask, dtype=bool)
        if oot.shape != (n,):
            raise ValueError("oot_mask must have same length as metrics.")
        fit_rows = finite_rows & oot
    else:
        fit_rows = finite_rows

    if np.sum(fit_rows) < max(10, X.shape[1] + 2):
        fit_rows = finite_rows
    if np.sum(fit_rows) < max(10, X.shape[1] + 2):
        raise ValueError("Not enough valid rows for diagnostics.")

    # Correlation matrix over the selected fit rows.
    corr = np.corrcoef(X[fit_rows].T)

    # Standardize using fit rows, then run SVD PCA.
    mu = np.nanmean(X[fit_rows], axis=0)
    sig = np.nanstd(X[fit_rows], axis=0)
    sig = np.where(sig > 0, sig, 1.0)
    Z = (X - mu[None, :]) / sig[None, :]
    Z[~finite_rows] = np.nan

    Zfit = Z[fit_rows]
    U, S, Vt = np.linalg.svd(Zfit, full_matrices=False)
    var = (S * S) / max(1, (Zfit.shape[0] - 1))
    varfrac = var / np.sum(var)

    k = min(int(n_components), Vt.shape[0])
    scores = np.full((n, k), np.nan, dtype=float)
    scores[finite_rows] = Z[finite_rows] @ Vt[:k].T

    if white_key in keys:
        iref = keys.index(white_key)
        corr_to_white = corr[iref]
    else:
        corr_to_white = np.full(len(keys), np.nan, dtype=float)

    return {
        "keys": keys,
        "corr": corr,
        "corr_to_white": corr_to_white,
        "fit_rows": fit_rows,
        "finite_rows": finite_rows,
        "scores": scores,
        "components": Vt[:k],
        "explained_variance_fraction": varfrac[:k],
    }


def plot_multimetric_correlation_and_pca(
    diag,
    white_light_norm=None,
    max_labels=20,
):
    """Plot a correlation heatmap, white-light correlations, and PCA summary."""
    keys = list(diag["keys"])
    corr = np.asarray(diag["corr"], dtype=float)
    cwhite = np.asarray(diag["corr_to_white"], dtype=float)
    evr = np.asarray(diag["explained_variance_fraction"], dtype=float)
    scores = np.asarray(diag["scores"], dtype=float)

    # Limit label clutter.
    if len(keys) > int(max_labels):
        keep = list(range(int(max_labels)))
        keys_plot = [keys[i] for i in keep]
        corr_plot = corr[np.ix_(keep, keep)]
        cwhite_plot = cwhite[keep]
    else:
        keys_plot = keys
        corr_plot = corr
        cwhite_plot = cwhite

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    im = ax.imshow(corr_plot, vmin=-1.0, vmax=1.0, cmap="coolwarm", origin="lower", aspect="auto")
    ax.set_title("Metric Correlation Matrix")
    ax.set_xticks(np.arange(len(keys_plot)))
    ax.set_yticks(np.arange(len(keys_plot)))
    ax.set_xticklabels(keys_plot, rotation=90, fontsize=7)
    ax.set_yticklabels(keys_plot, fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson r")

    ax = axes[0, 1]
    order = np.argsort(np.abs(cwhite_plot))[::-1]
    ax.bar(np.arange(len(order)), cwhite_plot[order], color="tab:blue", alpha=0.8)
    ax.axhline(0.0, color="k", lw=1, ls=":")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([keys_plot[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylabel("Corr with white_light_norm")
    ax.set_title("Absolute Correlation Ranking")

    ax = axes[1, 0]
    ax.plot(np.arange(1, evr.size + 1), 100.0 * evr, marker="o")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance [%]")
    ax.set_title("PCA Explained Variance")
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[1, 1]
    if scores.shape[1] >= 2:
        finite = np.isfinite(scores[:, 0]) & np.isfinite(scores[:, 1])
        if white_light_norm is not None:
            wl = np.asarray(white_light_norm, dtype=float)
            if wl.shape == scores[:, 0].shape:
                sc = ax.scatter(scores[finite, 0], scores[finite, 1], c=wl[finite], s=14, cmap="viridis", alpha=0.8)
                fig.colorbar(sc, ax=ax, label="white_light_norm")
            else:
                ax.scatter(scores[finite, 0], scores[finite, 1], s=14, alpha=0.8)
        else:
            ax.scatter(scores[finite, 0], scores[finite, 1], s=14, alpha=0.8)
        ax.set_xlabel("PC1 score")
        ax.set_ylabel("PC2 score")
        ax.set_title("PC1 vs PC2")
    else:
        ax.text(0.5, 0.5, "Need >=2 PCs", ha="center", va="center")
        ax.set_axis_off()

    plt.show()