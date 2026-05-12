import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from numba import jit, njit, prange
from tqdm import trange
from astropy.io import fits


@jit(nopython=True)
def nonlincor(f, px, py, nonlin_coeff):
    fcor = (
        nonlin_coeff[0, px, py]
        + nonlin_coeff[1, px, py] * f
        + nonlin_coeff[2, px, py] * f**2
        + nonlin_coeff[3, px, py] * f**3
        + nonlin_coeff[4, px, py] * f**4
        + nonlin_coeff[5, px, py] * f**5
    )
    return fcor


def plot_image(image, nstdp=1, nstdm=0.1, aspect=7):
    image_min = np.min(image)
    image_plot = image - image_min + 1

    cmean = np.median(image)
    cstd = np.std(image)

    vmax = cmean - image_min + cstd * nstdp + 1
    vmin = max(cmean - image_min - cstd * nstdm, 1)
    print(cmean, cstd, image_min, vmin, vmax)
    print(np.std(image[4:40, 4:40]))

    plt.figure(figsize=(10, 10))
    plt.imshow(image_plot, norm=LogNorm(vmin=vmin, vmax=vmax), aspect=aspect)
    plt.xlabel("X axis")
    plt.ylabel("Y-axis")
    plt.show()


@njit(fastmath=True)
def num_to_binary(number, size_of_bin=31):
    out = np.zeros(size_of_bin)
    num = number
    index = 31

    for _ in prange(size_of_bin):
        float_divide = num // 2
        divide = num / 2
        if float_divide != divide:
            out[index] = 1

        num = float_divide
        index -= 1
        if index == -1 or float_divide == 0:
            break

    return out


@jit(nopython=True)
def calc_refpix(scidata_array, dq_array, sat, bias, nrefcol=4):
    ns1 = scidata_array.shape[0]
    nr = scidata_array.shape[1]
    n1 = scidata_array.shape[2]
    n2 = scidata_array.shape[3]

    refpix = np.zeros((ns1, nr))

    left_n = nrefcol
    if left_n > n2:
        left_n = n2

    right_start = n2 - nrefcol
    if right_start < 0:
        right_start = 0

    for ni in range(ns1):
        for inr in range(nr):
            refpix_sum = 0.0
            refpix_n = 0

            for i in range(n1):
                sat_i = i + 1792

                for j in range(left_n):
                    if (dq_array[ni, i, j] & 1) == 0:
                        val = scidata_array[ni, inr, i, j] - bias[i, j]
                        if val < sat[sat_i, j]:
                            refpix_sum += val
                            refpix_n += 1

                for j in range(right_start, n2):
                    if j >= left_n and (dq_array[ni, i, j] & 1) == 0:
                        val = scidata_array[ni, inr, i, j] - bias[i, j]
                        if val < sat[sat_i, j]:
                            refpix_sum += val
                            refpix_n += 1

            if refpix_n > 0:
                refpix[ni, inr] = refpix_sum / refpix_n

    return refpix


def load_scidata_and_dq_with_groupdq(filename, workdir, data_ext, prodata_ext):
    """Load SCI, rateints DQ, optional GROUPDQ, and integration times.

    Parameters
    ----------
    filename : list[str]
        Basename list without suffixes.
    workdir : str
        Directory prefix for files.
    data_ext : str
        Uncal suffix, e.g. '_nis_uncal.fits'.
    prodata_ext : str
        Rateints suffix, e.g. '_nis_rateints.fits'.

    Returns
    -------
    scidata_array : np.ndarray
    dq_array : np.ndarray
    groupdq_array : np.ndarray | None
    int_times_list : list[np.ndarray]
    frmdivsr : int
    nframes : int
    """
    scidata_list = []
    dq_list = []
    groupdq_list = []
    int_times_list = []
    frmdivsr = -1
    nframes = -1

    for f in filename:
        with fits.open(workdir + f + data_ext) as hdulist:
            scidata_list.append(hdulist["SCI"].data.astype(float))

            if "INT_TIMES" in hdulist:
                int_times_list.append(hdulist["INT_TIMES"].data)
            else:
                int_times_list.append(hdulist[3].data)

            frmdivsr = hdulist[0].header.get("FRMDIVSR", frmdivsr)
            nframes = hdulist[0].header.get("NFRAMES", nframes)

            if "GROUPDQ" in hdulist:
                groupdq_list.append(hdulist["GROUPDQ"].data.astype(np.int32))

        with fits.open(workdir + f + prodata_ext) as hdulist:
            if "DQ" in hdulist:
                dq_list.append(hdulist["DQ"].data.astype(np.int32))
            else:
                dq_list.append(hdulist[3].data.astype(np.int32))

    scidata_array = np.concatenate(scidata_list, axis=0)
    dq_array = np.concatenate(dq_list, axis=0)

    if len(groupdq_list) == len(filename):
        groupdq_array = np.concatenate(groupdq_list, axis=0)
    else:
        groupdq_array = None

    return scidata_array, dq_array, groupdq_array, int_times_list, frmdivsr, nframes


@jit(nopython=True)
def stdev2(pts, mean):
    ep = 0.0
    var = 0.0
    npt = len(pts)
    for i in range(npt):
        s = pts[i] - mean
        ep = ep + s
        p = s * s
        var = var + p

    var = (var - ep**2 / npt) / (npt - 1)

    return np.sqrt(var)


@jit(nopython=True)
def lfit(npt, x, y, sig, flag, dydxmean):
    S = 0.0
    Sx = 0.0
    Sxx = 0.0
    Sy = 0.0
    Sxy = 0.0

    dy = np.zeros(npt)

    for i in range(npt):
        if flag[i] == 1:
            dy1 = y[i + 1] - y[i]
            for j in range(i, npt):
                dy[j + 1] += dy1 - dydxmean

    for i in range(npt):
        dsig2 = 1.0 / (sig[i] * sig[i])
        S += dsig2
        Sx += x[i] * dsig2
        Sxx += x[i] * x[i] * dsig2
        Sy += (y[i] - dy[i]) * dsig2
        Sxy += x[i] * (y[i] - dy[i]) * dsig2

    delta = S * Sxx - (Sx * Sx)
    if delta != 0:
        zpt = ((Sxx * Sy) - (Sx * Sxy)) / delta
        slope = ((S * Sxy) - (Sx * Sy)) / delta
    else:
        zpt = 0
        slope = 0

    return zpt, slope


@jit(nopython=True)
def r2s_test(scidata, dq, nrsatmap, bpix=-1.0e30, sigcut=2.0):
    nr = scidata.shape[0]
    n1 = scidata.shape[1]
    n2 = scidata.shape[2]

    x = np.zeros(nr)
    y = np.zeros(nr)
    ysig = np.zeros(nr)
    rflag = np.zeros(nr, dtype=np.int32)
    dydx = np.zeros(nr - 1)
    dydx_cut = np.zeros(nr - 1)

    zpt = np.zeros((n1, n2))
    stdimage = np.zeros((n1, n2))
    image = np.zeros((n1, n2))
    bpixmap = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            if (dq[i, j] & 1) == 0:
                npt = 0

                for k in range(nrsatmap[i, j]):
                    if scidata[k, i, j] > bpix:
                        npt += 1
                        x[npt - 1] = float(npt)
                        y[npt - 1] = scidata[k, i, j] * 1.0
                        ysig[npt - 1] = np.sqrt(np.abs(scidata[k, i, j])) + 1.0e-5
                        rflag[npt - 1] = 0

                if npt >= 3:
                    ndydx = npt - 1
                    for k in range(ndydx):
                        dydx[k] = (y[k + 1] - y[k]) / (x[k + 1] - x[k])

                    mean = np.median(dydx[0:ndydx])
                    std = stdev2(dydx[0:ndydx], mean)

                    zpt[i, j], image[i, j] = lfit(npt, x, y, ysig, rflag, mean)

                    icut = 0
                    icut_old = -1
                    while icut != icut_old:
                        npt_cut = 0
                        icut_old = icut
                        icut = 0
                        for k in range(ndydx):
                            if np.abs(dydx[k] - mean) < sigcut * std:
                                npt_cut += 1
                                dydx_cut[npt_cut - 1] = dydx[k]
                                rflag[k] = 0
                            else:
                                icut += 1
                                rflag[k] = 1

                        if npt_cut > 1:
                            mean = np.mean(dydx_cut[0:npt_cut])
                            std = np.std(dydx_cut[0:npt_cut])
                            zpt[i, j], image[i, j] = lfit(npt, x, y, ysig, rflag, mean)

                    stdimage[i, j] = std

                elif npt == 0:
                    zpt[i, j] = 0.0
                    stdimage[i, j] = 0.0
                    image[i, j] = 0.0

                else:
                    zpt[i, j] = scidata[0, i, j] * 1.0
                    stdimage[i, j] = 0.0
                    image[i, j] = (scidata[nrsatmap[i, j] - 1, i, j]) * nr / nrsatmap[i, j]
            else:
                bpixmap[i, j] = 1

    # Allocate once and reuse to avoid repeated allocations in the bad-pixel fill loop.
    max_neighbors = n1 * n2
    pixels = np.zeros(max_neighbors)

    for i in range(n1):
        for j in range(n2):
            if bpixmap[i, j] > 0:
                ng = 1
                icor = 0
                ng_max = n1 if n1 > n2 else n2

                # Guard against pathological all-bad regions to prevent an infinite loop.
                while icor == 0 and ng <= ng_max:

                    k = 0
                    i_start = 0 if i - ng < 0 else i - ng
                    i_stop = n1 if i + ng + 1 > n1 else i + ng + 1
                    j_start = 0 if j - ng < 0 else j - ng
                    j_stop = n2 if j + ng + 1 > n2 else j + ng + 1

                    for i2 in range(i_start, i_stop):
                        for j2 in range(j_start, j_stop):
                            if bpixmap[i2, j2] < 1:
                                pixels[k] = image[i2, j2]
                                k += 1
                    if k > 0:
                        image[i, j] = np.median(pixels[:k])
                        icor = 1
                    else:
                        ng += 1

                if icor == 0:
                    image[i, j] = 0.0

    return zpt, stdimage, image, bpixmap


@jit(nopython=True)
def _linear_fit_weighted(x, y, npt):
    S = 0.0
    Sx = 0.0
    Sxx = 0.0
    Sy = 0.0
    Sxy = 0.0

    for i in range(npt):
        sig = np.sqrt(np.abs(y[i])) + 1.0e-5
        w = 1.0 / (sig * sig)
        S += w
        Sx += x[i] * w
        Sxx += x[i] * x[i] * w
        Sy += y[i] * w
        Sxy += x[i] * y[i] * w

    delta = S * Sxx - Sx * Sx
    if delta == 0.0:
        return 0.0, 0.0

    zpt = (Sxx * Sy - Sx * Sxy) / delta
    slope = (S * Sxy - Sx * Sy) / delta
    return zpt, slope


@jit(nopython=True)
def r2s_test_groupdq(
    scidata,
    groupdq,
    bpix=-1.0e30,
    do_not_use_bit=1,
    saturated_bit=2,
    jump_bit=4,
):
    nr = scidata.shape[0]
    n1 = scidata.shape[1]
    n2 = scidata.shape[2]

    x = np.zeros(nr)
    y = np.zeros(nr)
    dydx = np.zeros(nr - 1)

    zpt = np.zeros((n1, n2))
    stdimage = np.zeros((n1, n2))
    image = np.zeros((n1, n2))
    bpixmap = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            best_n = 0
            best_zpt = 0.0
            best_slope = 0.0
            best_std = 0.0

            nseg = 0
            saw_saturation = 0

            for k in range(nr):
                dqk = groupdq[k, i, j]

                if (dqk & jump_bit) != 0 and nseg >= 2:
                    zpt_seg, slope_seg = _linear_fit_weighted(x, y, nseg)

                    if nseg > best_n:
                        best_n = nseg
                        best_zpt = zpt_seg
                        best_slope = slope_seg
                        if nseg >= 3:
                            nd = nseg - 1
                            for kk in range(nd):
                                dydx[kk] = y[kk + 1] - y[kk]
                            best_std = np.std(dydx[:nd])
                        else:
                            best_std = 0.0

                    nseg = 0

                if (dqk & do_not_use_bit) != 0:
                    if nseg >= 2:
                        zpt_seg, slope_seg = _linear_fit_weighted(x, y, nseg)

                        if nseg > best_n:
                            best_n = nseg
                            best_zpt = zpt_seg
                            best_slope = slope_seg
                            if nseg >= 3:
                                nd = nseg - 1
                                for kk in range(nd):
                                    dydx[kk] = y[kk + 1] - y[kk]
                                best_std = np.std(dydx[:nd])
                            else:
                                best_std = 0.0

                    nseg = 0

                    if (dqk & saturated_bit) != 0:
                        saw_saturation = 1
                        break

                    continue

                if (dqk & saturated_bit) != 0:
                    if nseg >= 2:
                        zpt_seg, slope_seg = _linear_fit_weighted(x, y, nseg)

                        if nseg > best_n:
                            best_n = nseg
                            best_zpt = zpt_seg
                            best_slope = slope_seg
                            if nseg >= 3:
                                nd = nseg - 1
                                for kk in range(nd):
                                    dydx[kk] = y[kk + 1] - y[kk]
                                best_std = np.std(dydx[:nd])
                            else:
                                best_std = 0.0

                    saw_saturation = 1
                    break

                val = scidata[k, i, j]
                if val > bpix:
                    x[nseg] = float(k + 1)
                    y[nseg] = val
                    nseg += 1

            if nseg >= 2:
                zpt_seg, slope_seg = _linear_fit_weighted(x, y, nseg)

                if nseg > best_n:
                    best_n = nseg
                    best_zpt = zpt_seg
                    best_slope = slope_seg
                    if nseg >= 3:
                        nd = nseg - 1
                        for kk in range(nd):
                            dydx[kk] = y[kk + 1] - y[kk]
                        best_std = np.std(dydx[:nd])
                    else:
                        best_std = 0.0

            if best_n >= 2:
                zpt[i, j] = best_zpt
                image[i, j] = best_slope
                stdimage[i, j] = best_std
            elif best_n == 1:
                zpt[i, j] = y[0]
                image[i, j] = 0.0
                stdimage[i, j] = 0.0
            else:
                bpixmap[i, j] = 1.0
                if saw_saturation == 1:
                    zpt[i, j] = 0.0
                    image[i, j] = 0.0
                    stdimage[i, j] = 0.0

    max_neighbors = n1 * n2
    pixels = np.zeros(max_neighbors)

    for i in range(n1):
        for j in range(n2):
            if bpixmap[i, j] > 0:
                ng = 1
                icor = 0
                ng_max = n1 if n1 > n2 else n2

                while icor == 0 and ng <= ng_max:
                    k = 0
                    i_start = 0 if i - ng < 0 else i - ng
                    i_stop = n1 if i + ng + 1 > n1 else i + ng + 1
                    j_start = 0 if j - ng < 0 else j - ng
                    j_stop = n2 if j + ng + 1 > n2 else j + ng + 1

                    for i2 in range(i_start, i_stop):
                        for j2 in range(j_start, j_stop):
                            if bpixmap[i2, j2] < 1:
                                pixels[k] = image[i2, j2]
                                k += 1

                    if k > 0:
                        image[i, j] = np.median(pixels[:k])
                        icor = 1
                    else:
                        ng += 1

                if icor == 0:
                    image[i, j] = 0.0

    return zpt, stdimage, image, bpixmap


@jit(nopython=True)
def zpt_create(zpt, image, thres=200):
    n1 = zpt.shape[0]
    n2 = zpt.shape[1]

    zpt_temp = np.copy(zpt)

    zpt_row = np.zeros((n1))
    for i in range(n1):
        zpt_row[i] = np.median((zpt_temp[i, 4 : n2 - 4])[image[i, 4 : n2 - 4] < thres])
        zpt_temp[i, :] -= zpt_row[i]

    zpt_col = np.zeros((n2))
    for j in range(n2):
        zpt_col[j] = np.median((zpt_temp[: n1 - 4, j])[image[: n1 - 4, j] < thres])

    zpt_new = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            zpt_new[i, j] = zpt_row[i] + zpt_col[j]

    return zpt_new


@jit(nopython=True)
def ramp_stack(scidata_sb_ref_lin):
    nr = scidata_sb_ref_lin.shape[1]
    n1 = scidata_sb_ref_lin.shape[2]
    n2 = scidata_sb_ref_lin.shape[3]

    scidata_sb_ref_lin_nrmed = np.zeros((nr, n1, n2))

    for k in range(nr):
        for i in range(n1):
            for j in range(n2):
                scidata_sb_ref_lin_nrmed[k, i, j] = np.median(scidata_sb_ref_lin[:, k, i, j])

    return scidata_sb_ref_lin_nrmed


@jit(nopython=True)
def calc_sb_ref_lin(scidata_array, dq_array, refpix, nonlin_coeff, sat, bias):
    return calc_sb_ref_lin_with_progress(scidata_array, dq_array, refpix, nonlin_coeff, sat, bias)


@njit
def _build_nrsatmap_single(scidata_int, sat, detector_row0, ng):
    nr = scidata_int.shape[0]
    n1 = scidata_int.shape[1]
    n2 = scidata_int.shape[2]

    nrsat = np.zeros((n1, n2), dtype=np.int32)

    for i in range(n1):
        for j in range(n2):
            isat = 0
            for k in range(nr):
                if (scidata_int[k, i, j] < sat[i + detector_row0, j]) and (isat == 0):
                    nrsat[i, j] = k + 1
                elif (scidata_int[k, i, j] >= sat[i + detector_row0, j]) and (isat == 0):
                    for i2 in range(max(0, i - ng), min(n1, i + ng + 1)):
                        for j2 in range(max(0, j - ng), min(n2, j + ng + 1)):
                            nrsat[i2, j2] = k
                    isat = 1

    return nrsat


@njit
def _apply_bias_ref_nonlin_single(scidata_int, refpix_int, nonlin_coeff, bias, detector_row0):
    nr = scidata_int.shape[0]
    n1 = scidata_int.shape[1]
    n2 = scidata_int.shape[2]

    out = np.zeros((nr, n1, n2))

    for inr in range(nr):
        ref = refpix_int[inr]
        for i in range(n1):
            for j in range(n2):
                val = scidata_int[inr, i, j] - bias[i, j] - ref
                out[inr, i, j] = nonlincor(val, i + detector_row0, j, nonlin_coeff)

    return out


def calc_sb_ref_lin_with_progress(
    scidata_array,
    dq_array,
    refpix,
    nonlin_coeff,
    sat,
    bias,
    detector_row0=1792,
    bleed_ng=1,
    thres=500.0,
    use_tqdm=True,
):
    ns1 = scidata_array.shape[0]
    nr = scidata_array.shape[1]
    n1 = scidata_array.shape[2]
    n2 = scidata_array.shape[3]

    if detector_row0 + n1 > sat.shape[0]:
        raise ValueError("detector_row0 with current subarray height exceeds saturation reference dimensions")

    scidata_sb_ref_lin = np.zeros((ns1, nr, n1, n2))
    nrsatmap = np.zeros((ns1, n1, n2), dtype=np.int32)

    iter_1 = trange(ns1, desc="calc_sb_ref_lin phase 1/2", unit="int") if use_tqdm else range(ns1)
    for ins1 in iter_1:
        nrsatmap[ins1] = _build_nrsatmap_single(scidata_array[ins1], sat, detector_row0, bleed_ng)

        scidata_sb_ref_lin[ins1] = _apply_bias_ref_nonlin_single(
            scidata_array[ins1], refpix[ins1], nonlin_coeff, bias, detector_row0
        )

        zpt, stdimage, image_linfit, bpixmap = r2s_test(scidata_sb_ref_lin[ins1], dq_array[ins1], nrsatmap[ins1])
        zpt_new = zpt_create(zpt, image_linfit)
        scidata_sb_ref_lin[ins1] -= zpt_new

    scidata_sb_ref_lin_nrmed = ramp_stack(scidata_sb_ref_lin)

    iter_2 = trange(ns1, desc="calc_sb_ref_lin phase 2/2", unit="int") if use_tqdm else range(ns1)
    for ins1 in iter_2:
        for k in range(nr):
            diff_image = scidata_sb_ref_lin[ins1, k] - scidata_sb_ref_lin_nrmed[k]
            onedf_col = np.zeros(n2)

            for j in range(n2):
                vals = diff_image[4 : n1 - 4, j]
                mask = scidata_sb_ref_lin[ins1, k, 4 : n1 - 4, j] < thres
                if np.any(mask):
                    onedf_col[j] = np.median(vals[mask])
                else:
                    onedf_col[j] = 0.0

            scidata_sb_ref_lin[ins1, k] -= onedf_col[np.newaxis, :]

    return scidata_sb_ref_lin, nrsatmap


@jit(nopython=True)
def stackimage(clean_data):
    n1 = clean_data.shape[1]
    n2 = clean_data.shape[2]

    deepstack = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            deepstack[i, j] = np.median(clean_data[:, i, j])

    return deepstack


@jit(nopython=True)
def cr_cor(clean_diff, clean_data, cr_thres=5.0):
    clean_diff_cr = np.copy(clean_diff)
    clean_data_cr = np.copy(clean_data)
    clean_diff_in = np.copy(clean_diff)
    clean_data_in = np.copy(clean_data)
    npt = clean_diff.shape[0]
    n1 = clean_diff.shape[1]
    n2 = clean_diff.shape[2]

    max_neighbors = 4
    diff_neighbors = np.zeros(max_neighbors)
    data_neighbors = np.zeros(max_neighbors)

    for i in range(n1):
        for j in range(n2):
            mean, std = _median_abs_dev_sigma(clean_diff_in[:, i, j])
            if std <= 0.0:
                continue

            for k in range(npt):
                if np.abs(clean_diff_in[k, i, j] - mean) > cr_thres * std:
                    nnb = 0
                    radius = 1

                    while radius < npt and nnb < max_neighbors:
                        k1 = k - radius
                        if k1 >= 0 and np.abs(clean_diff_in[k1, i, j] - mean) <= cr_thres * std:
                            diff_neighbors[nnb] = clean_diff_in[k1, i, j]
                            data_neighbors[nnb] = clean_data_in[k1, i, j]
                            nnb += 1
                            if nnb >= max_neighbors:
                                break

                        k2 = k + radius
                        if k2 < npt and np.abs(clean_diff_in[k2, i, j] - mean) <= cr_thres * std:
                            diff_neighbors[nnb] = clean_diff_in[k2, i, j]
                            data_neighbors[nnb] = clean_data_in[k2, i, j]
                            nnb += 1

                        radius += 1

                    if nnb > 0:
                        clean_diff_cr[k, i, j] = np.median(diff_neighbors[:nnb])
                        clean_data_cr[k, i, j] = np.median(data_neighbors[:nnb])

    return clean_diff_cr, clean_data_cr


def _resample_axis_centers(values, nout):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("Axis values must be one-dimensional")
    if values.size == 0 or nout <= 0:
        raise ValueError("Axis values must be non-empty and nout must be positive")

    def _fill_nonfinite_1d(arr):
        arr = np.asarray(arr, dtype=float).copy()
        finite = np.isfinite(arr)
        if np.all(finite):
            return arr

        if not np.any(finite):
            return np.arange(arr.size, dtype=float)

        finite_idx = np.flatnonzero(finite)
        if finite_idx.size == 1:
            arr[:] = arr[finite_idx[0]]
            return arr

        all_idx = np.arange(arr.size, dtype=float)
        arr[~finite] = np.interp(all_idx[~finite], all_idx[finite], arr[finite])
        return arr

    if values.size == nout:
        return _fill_nonfinite_1d(values)

    groups = np.array_split(values, nout)
    centers = np.empty(nout, dtype=float)
    for index, group in enumerate(groups):
        finite_group = group[np.isfinite(group)]
        if finite_group.size > 0:
            centers[index] = np.median(finite_group)
        else:
            centers[index] = np.nan
    return _fill_nonfinite_1d(centers)


def _centers_to_edges(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1:
        raise ValueError("Centers must be one-dimensional")
    if centers.size == 0:
        raise ValueError("Centers must be non-empty")
    if centers.size == 1:
        delta = 0.5
        return np.array([centers[0] - delta, centers[0] + delta], dtype=float)

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def plotextract_long(tspec, vmin, vmax, time, wavesol, c1=0, c2=None, sig=3.0):
    cmap = plt.get_cmap("plasma")

    if c2 is None:
        c2 = tspec.shape[1]

    tspec_plot = np.asarray(tspec, dtype=float)
    time_centers = _resample_axis_centers(time, tspec_plot.shape[0])
    time_hours = 24.0 * (time_centers - time_centers[0])
    wave_centers = _resample_axis_centers(wavesol, tspec_plot.shape[1])
    time_edges = _centers_to_edges(time_hours)
    wave_edges = _centers_to_edges(wave_centers)

    figwidth = 12
    figheight = 12
    xmin = wave_edges[0]
    xmax = wave_edges[-1]
    ymin = time_edges[0]
    ymax = time_edges[-1]

    fig = plt.figure(figsize=(figwidth, figheight))
    ax = plt.axes()
    ax.tick_params(direction="in", which="major", bottom=True, top=True, left=True, right=True, length=10, width=2)
    ax.tick_params(direction="in", which="minor", bottom=True, top=True, left=True, right=True, length=4, width=2)

    ax.set_xlabel("Wavelength ($\\mu$m)")
    ax.set_ylabel("Time (hours)")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())

    if np.abs(vmin - vmax) < 0.0001:
        vmin = max([0.01, np.median(tspec_plot[:, c1:c2]) - sig * np.std(tspec_plot[:, c1:c2])])
        vmax = np.median(tspec_plot[:, c1:c2]) + sig * np.std(tspec_plot[:, c1:c2])
        print(vmin, vmax)

    ax.pcolormesh(wave_edges, time_edges, tspec_plot, shading="auto", zorder=1, vmin=vmin, vmax=vmax, cmap=cmap)

    return fig, ax


def plotimage(deepstack, sig=3.0, c1=0, c2=None, cmap_name="viridis"):
    deepstack = np.asarray(deepstack, dtype=float)
    if deepstack.ndim != 2:
        raise ValueError("deepstack must be 2D")

    if c2 is None:
        c2 = deepstack.shape[1]
    c1 = max(0, int(c1))
    c2 = min(deepstack.shape[1], int(c2))
    if c2 <= c1:
        c1, c2 = 0, deepstack.shape[1]

    figwidth = 16
    figheight = figwidth * deepstack.shape[0] / deepstack.shape[1]
    xmin = 1
    xmax = deepstack.shape[1]
    ymin = 1
    ymax = deepstack.shape[0]

    fig = plt.figure(figsize=(figwidth, figheight))
    ax = plt.axes()
    ax.tick_params(direction="in", which="major", bottom=True, top=True, left=True, right=True, length=10, width=2)
    ax.tick_params(direction="in", which="minor", bottom=True, top=True, left=True, right=True, length=4, width=2)

    ax.set_xlabel("Column (pixels)")
    ax.set_ylabel("Row (pixels)")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    clip = deepstack[:, c1:c2]
    vmin = max(0.1, np.median(clip) - sig * np.std(clip))
    vmax = np.median(clip) + sig * np.std(clip)

    x_axis = np.linspace(xmin, xmax, xmax)
    y_axis = np.linspace(ymin, ymax, ymax)
    ax.pcolormesh(x_axis, y_axis, deepstack, zorder=1, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=plt.get_cmap(cmap_name))

    return fig, ax


def plotextract(tspec, vmin, vmax, sig=3.0, c1=0, c2=None, cmap_name="magma"):
    tspec = np.asarray(tspec, dtype=float)
    if tspec.ndim != 2:
        raise ValueError("tspec must be 2D")

    if c2 is None:
        c2 = tspec.shape[1]
    c1 = max(0, int(c1))
    c2 = min(tspec.shape[1], int(c2))
    if c2 <= c1:
        c1, c2 = 0, tspec.shape[1]

    figwidth = 16
    figheight = figwidth * tspec.shape[0] / tspec.shape[1]
    xmin = 1
    xmax = tspec.shape[1]
    ymin = 1
    ymax = tspec.shape[0]

    fig = plt.figure(figsize=(figwidth, figheight))
    ax = plt.axes()
    ax.tick_params(direction="in", which="major", bottom=True, top=True, left=True, right=True, length=10, width=2)
    ax.tick_params(direction="in", which="minor", bottom=True, top=True, left=True, right=True, length=4, width=2)

    ax.set_xlabel("Spectral Channel")
    ax.set_ylabel("Int #")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if np.abs(vmin - vmax) < 0.000001:
        clip = tspec[:, c1:c2]
        vmin = max(0.01, np.median(clip) - sig * np.std(clip))
        vmax = np.median(clip) + sig * np.std(clip)
        print(vmin, vmax)

    x_axis = np.linspace(xmin, xmax, xmax)
    y_axis = np.linspace(ymin, ymax, ymax)
    ax.pcolormesh(x_axis, y_axis, tspec, zorder=1, vmin=vmin, vmax=vmax, cmap=plt.get_cmap(cmap_name))

    return fig, ax


@jit(nopython=True)
def gpsf_model(pars, x):
    xcen = pars[0]
    sig = pars[1]
    amp = pars[2]
    cap = pars[3]

    npt = x.shape[0]
    model = np.zeros(npt)
    for i in range(npt):
        model1 = amp * np.exp(-((x[i] - xcen) * (x[i] - xcen)) / (2.0 * sig * sig))
        if model1 >= cap:
            model[i] = cap
        else:
            model[i] = model1

    return model


@jit(nopython=True)
def lorentz_psf_model(pars, x):
    xcen = pars[0]
    sig = pars[1]
    amp = pars[2]
    cap = pars[3]

    npt = x.shape[0]
    model = np.zeros(npt)
    for i in range(npt):
        model1 = amp * (sig / (((x[i] - xcen) * (x[i] - xcen)) + (0.5 * sig) * (0.5 * sig)))
        if model1 >= cap:
            model[i] = cap
        else:
            model[i] = model1

    return model


@jit(nopython=True)
def fieldmodel_psf(fitpars, x, colstart, tr, photap):
    ftest = np.min(fitpars)
    if ftest < 0:
        return np.zeros(x.shape[0])

    model = np.zeros(x.shape[0])

    if (tr[colstart, 0] > 0) and (tr[colstart, 0] - photap < 350):
        pars = np.array([tr[colstart, 0], fitpars[0], fitpars[1], fitpars[2]])
        pars2 = np.array([tr[colstart, 0], fitpars[12], fitpars[13], fitpars[13]])
        model += gpsf_model(pars, x) + lorentz_psf_model(pars2, x)

    if (tr[colstart, 1] > 0) and (tr[colstart, 1] - photap < 350):
        pars = np.array([tr[colstart, 1], fitpars[0], fitpars[3], fitpars[4]])
        pars2 = np.array([tr[colstart, 1], fitpars[12], fitpars[14], fitpars[14]])
        model += gpsf_model(pars, x) + lorentz_psf_model(pars2, x)

    if (tr[colstart, 2] > 0) and (tr[colstart, 2] - photap < 350):
        pars = np.array([tr[colstart, 2] + 2, fitpars[0], fitpars[5], fitpars[6]])
        pars2 = np.array([tr[colstart, 2], fitpars[12], fitpars[15], fitpars[15]])
        model += gpsf_model(pars, x) + lorentz_psf_model(pars2, x)

    if (tr[colstart, 3] > 0) and (tr[colstart, 3] - photap < 350):
        pars = np.array([tr[colstart, 3], fitpars[0], fitpars[7], fitpars[8]])
        pars2 = np.array([tr[colstart, 3], fitpars[12], fitpars[16], fitpars[16]])
        model += gpsf_model(pars, x) + lorentz_psf_model(pars2, x)

    if (tr[colstart, 4] > 0) and (tr[colstart, 4] - photap < 300):
        pars = np.array([tr[colstart, 4], fitpars[0], fitpars[9], fitpars[10]])
        pars2 = np.array([tr[colstart, 4], fitpars[12], fitpars[17], fitpars[17]])
        model += gpsf_model(pars, x) + lorentz_psf_model(pars2, x)

    model += fitpars[11]
    return model


@jit(nopython=True)
def diffmodel_psf(fitpars, x, y, colstart, fitpars_init, ifitpars, tr, photap):
    fieldpars = np.copy(fitpars_init)
    nfitmax = fieldpars.shape[0]

    i = 0
    for j in range(nfitmax):
        if ifitpars[j] == 1:
            fieldpars[j] = fitpars[i]
            i += 1

    model = fieldmodel_psf(fieldpars, x, colstart, tr, photap)
    diff = (model - y) / np.sqrt(y)

    return diff


def plot_fieldmodel_psf(xdata, ydata, fitpars_ans, colstart, photap, ymin, ymax, tr, wavesol):
    fig = plt.figure(figsize=(20, 4))
    matplotlib.rcParams.update({"font.size": 20})
    ax = plt.axes()
    ax.plot(xdata, ydata, marker="o")

    if tr[colstart, 0] > 0:
        ans_tmp = np.copy(fitpars_ans)
        for index in [3, 5, 7, 9]:
            ans_tmp[index] = 0.0
        for index in [14, 15, 16, 17]:
            ans_tmp[index] = 0.0
        model = fieldmodel_psf(ans_tmp, xdata, colstart, tr, photap)
        ax.plot(xdata, model, color="red")
        rec = patches.Rectangle((tr[colstart, 0] - photap, ymin), 2 * photap, ymax - ymin, color="red", alpha=0.2)
        ax.add_patch(rec)

    if tr[colstart, 1] > 0:
        ans_tmp = np.copy(fitpars_ans)
        for index in [1, 5, 7, 9]:
            ans_tmp[index] = 0.0
        for index in [13, 15, 16, 17]:
            ans_tmp[index] = 0.0
        model = fieldmodel_psf(ans_tmp, xdata, colstart, tr, photap)
        ax.plot(xdata, model, color="orange")
        rec = patches.Rectangle((tr[colstart, 1] - photap, ymin), 2 * photap, ymax - ymin, color="orange", alpha=0.2)
        ax.add_patch(rec)

    if tr[colstart, 2] > 0:
        ans_tmp = np.copy(fitpars_ans)
        for index in [1, 3, 7, 9]:
            ans_tmp[index] = 0.0
        for index in [13, 14, 16, 17]:
            ans_tmp[index] = 0.0
        model = fieldmodel_psf(ans_tmp, xdata, colstart, tr, photap)
        ax.plot(xdata, model, color="green")
        rec = patches.Rectangle((tr[colstart, 2] - photap, ymin), 2 * photap, ymax - ymin, color="green", alpha=0.2)
        ax.add_patch(rec)

    if tr[colstart, 3] > 0:
        ans_tmp = np.copy(fitpars_ans)
        for index in [1, 3, 5, 9]:
            ans_tmp[index] = 0.0
        for index in [13, 14, 15, 17]:
            ans_tmp[index] = 0.0
        model = fieldmodel_psf(ans_tmp, xdata, colstart, tr, photap)
        ax.plot(xdata, model, color="blue")
        rec = patches.Rectangle((tr[colstart, 3] - photap, ymin), 2 * photap, ymax - ymin, color="blue", alpha=0.2)
        ax.add_patch(rec)

    if tr[colstart, 4] > 0:
        ans_tmp = np.copy(fitpars_ans)
        for index in [1, 3, 5, 7]:
            ans_tmp[index] = 0.0
        for index in [13, 14, 15, 16]:
            ans_tmp[index] = 0.0
        model = fieldmodel_psf(ans_tmp, xdata, colstart, tr, photap)
        ax.plot(xdata, model, color="cyan")
        rec = patches.Rectangle((tr[colstart, 4] - photap, ymin), 2 * photap, ymax - ymin, color="cyan", alpha=0.2)
        ax.add_patch(rec)

    print(fitpars_ans[0])
    print(fitpars_ans[1], fitpars_ans[2])
    print(fitpars_ans[3], fitpars_ans[4])
    print(fitpars_ans[5], fitpars_ans[6])
    print(fitpars_ans[7], fitpars_ans[8])
    print(fitpars_ans[9], fitpars_ans[10])
    print("zpt:", fitpars_ans[11], wavesol[0, colstart])
    print(fitpars_ans[12])
    print(fitpars_ans[13], fitpars_ans[14], fitpars_ans[15], fitpars_ans[16], fitpars_ans[17])
    ax.hlines(y=fitpars_ans[11], xmin=0, xmax=250)

    diff = fieldmodel_psf(fitpars_ans, xdata, colstart, tr, photap) - ydata
    ax.plot(xdata, diff + 10)

    ax.set_xlim(0, 250)
    ax.hlines(y=10, xmin=0, xmax=250, color="grey")
    ax.set_ylim(ymin, ymax)
    plt.yscale("log")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Counts")
    plt.show()


def default_psf_fitpars_init():
    return np.array(
        [
            8.0,
            800.0,
            300.0,
            50.0,
            23.0,
            200.0,
            60.0,
            1350.0,
            560.0,
            200.0,
            5.0,
            22.0,
            40.0,
            400.0,
            50.0,
            200.0,
            400.0,
            50.0,
        ],
        dtype=float,
    )


def _estimate_component_seed(ydata, trace_row, photap, background):
    x1 = max(int(np.floor(trace_row - photap)), 0)
    x2 = min(int(np.ceil(trace_row + photap)) + 1, ydata.shape[0])
    if x2 <= x1:
        return 0.0, 0.0

    local_profile = ydata[x1:x2] - background
    if local_profile.size == 0:
        return 0.0, 0.0

    peak_signal = float(np.max(local_profile))
    if peak_signal <= 0.0:
        return 0.0, 0.0

    positive_signal = local_profile[local_profile > 0.0]
    if positive_signal.size == 0:
        return peak_signal, 0.0

    return peak_signal, float(np.sum(positive_signal))


def fit_deepstack_background_and_contamination(
    deepstack,
    tr,
    skymap,
    fieldmodel=None,
    diffmodel=None,
    plot_fieldmodel=None,
    photap=16,
    ymin=10**0.5,
    ymax=10**4.5,
    fitpars_init=None,
    cstart=5,
    cend=None,
    plot_every=100,
    verbose=True,
    least_squares_func=None,
    wavesol=None,
):
    if least_squares_func is None:
        from scipy.optimize import least_squares

        least_squares_func = least_squares

    if cend is None:
        cend = deepstack.shape[1] - 5

    if fitpars_init is None:
        fitpars_init = default_psf_fitpars_init()

    if fieldmodel is None:
        def fieldmodel(fitpars, x, colstart):
            return fieldmodel_psf(fitpars, x, colstart, tr, photap)

    if diffmodel is None:
        def diffmodel(fitpars, x, y, colstart, fitpars_init, ifitpars):
            return diffmodel_psf(fitpars, x, y, colstart, fitpars_init, ifitpars, tr, photap)

    if plot_fieldmodel is None and wavesol is not None:
        def plot_fieldmodel(xdata, ydata, fitpars_ans, colstart, photap, ymin, ymax):
            return plot_fieldmodel_psf(xdata, ydata, fitpars_ans, colstart, photap, ymin, ymax, tr, wavesol)

    fitpars_init = np.asarray(fitpars_init, dtype=float).copy()
    if fitpars_init.shape != (18,):
        raise ValueError("fitpars_init must have shape (18,)")

    default_fitpars = default_psf_fitpars_init()
    component_param_map = (
        (1, 2, 13, 350.0),
        (3, 4, 14, 350.0),
        (5, 6, 15, 350.0),
        (7, 8, 16, 350.0),
        (9, 10, 17, 300.0),
    )

    ifitpars = np.ones(fitpars_init.shape[0], dtype=np.int32)
    bkg_model = np.zeros(deepstack.shape[1], dtype=float)
    contam_model = np.zeros((deepstack.shape[1], 5), dtype=float)

    for index in [9, 10, 17]:
        fitpars_init[index] = 0.0
        ifitpars[index] = 0

    nrows = min(250, deepstack.shape[0])
    xdata = np.arange(nrows, dtype=float)
    fitpars_ans = np.copy(fitpars_init)

    for colstart_init in range(cstart, cend):
        if colstart_init > cstart:
            fitpars_init = np.copy(fitpars_ans)

        j = 1
        for index in [4, 6, 8, 10]:
            if fitpars_init[index] < 200:
                if (index == 10) and (colstart_init > 1060):
                    ifitpars[16] = 1
                else:
                    fitpars_init[13 + j] = 0.0
                    ifitpars[13 + j] = 0
            else:
                ifitpars[13 + j] = 1
            j += 1

        fitpars_init[15] = 0.0
        ifitpars[15] = 0

        fitpars_init[11] = skymap[colstart_init]
        ifitpars[11] = 0

        if colstart_init == 450:
            fitpars_init[9] = 200.0
            ifitpars[9] = 1
            fitpars_init[10] = 50.0
            ifitpars[10] = 1

        if colstart_init > 1080:
            fitpars_init[9] = 0.0
            ifitpars[9] = 0
            fitpars_init[10] = 0.0
            ifitpars[10] = 0

        if colstart_init > 890:
            for index in [5, 6, 15]:
                fitpars_init[index] = 0.0
                ifitpars[index] = 0

        if colstart_init > 1480:
            fitpars_init[17] = 0.0
            ifitpars[17] = 0

        if 1690 <= colstart_init < 1710:
            fitpars_init[3] = 2000.0
            ifitpars[3] = 1
            fitpars_init[4] = 1000.0
            ifitpars[4] = 1

        ydata = np.array(deepstack[:nrows, colstart_init], dtype=float, copy=True)
        if np.any(ydata <= 0):
            ydata[ydata <= 0] = np.median(ydata)

        # Only intervene when a visible component has clearly collapsed
        # relative to the local column profile. This avoids forcing all
        # components toward shallow square-topped solutions.
        for component_index, (amp_index, cap_index, lorentz_index, row_limit) in enumerate(component_param_map):
            trace_row = tr[colstart_init, component_index]
            if (trace_row <= 0.0) or (trace_row - photap >= row_limit):
                continue

            peak_signal, _ = _estimate_component_seed(ydata, trace_row, photap, fitpars_init[11])
            if peak_signal <= 0.0:
                continue

            modeled_peak = min(fitpars_init[amp_index], fitpars_init[cap_index])
            collapsed_component = modeled_peak < 0.25 * peak_signal
            significant_signal = peak_signal > 0.75 * default_fitpars[cap_index]

            if collapsed_component and significant_signal:
                fitpars_init[amp_index] = max(fitpars_init[amp_index], min(peak_signal, 1.25 * default_fitpars[amp_index]))
                fitpars_init[cap_index] = max(fitpars_init[cap_index], min(peak_signal, default_fitpars[cap_index]))

            if (lorentz_index != 15) and collapsed_component and significant_signal and (ifitpars[lorentz_index] == 0):
                fitpars_init[lorentz_index] = max(fitpars_init[lorentz_index], default_fitpars[lorentz_index])
                ifitpars[lorentz_index] = 1

        fitpars = fitpars_init[ifitpars == 1]
        ans_fm = least_squares_func(diffmodel, fitpars, args=[xdata, ydata, colstart_init, fitpars_init, ifitpars], method="lm")

        fitpars_ans = np.copy(fitpars_init)
        fitpars_ans[ifitpars == 1] = ans_fm.x

        bkg_model[colstart_init] = fitpars_ans[11]

        j = 0
        for index in [1, 3, 5, 7, 9]:
            ans_tmp = np.copy(fitpars_ans)
            ans_tmp[index] = 0.0
            ans_tmp[11] = 0.0
            ans_tmp[13 + j] = 0.0
            model_contam = fieldmodel(ans_tmp, xdata, colstart_init)
            x1 = int(max(tr[colstart_init, j] - photap, 5))
            x2 = int(min(tr[colstart_init, j] + photap, nrows))
            contam_model[colstart_init, j] = np.sum(model_contam[x1:x2])
            j += 1

        if verbose and plot_fieldmodel is not None and plot_every > 0 and (colstart_init % plot_every == 0):
            print(colstart_init)
            print(contam_model[colstart_init])
            plot_fieldmodel(xdata, ydata, fitpars_ans, colstart_init, photap, ymin, ymax)

    return bkg_model, contam_model, fitpars_ans


@jit(nopython=True)
def apfluxex(image, tr, ap1, ap2, bkg_model, contam_model, tr_order=0, skycorr=True):
    nrows = image.shape[0]
    nch = image.shape[1]

    traceflux = np.zeros(nch)

    for i in range(nch):
        xcoo = tr[i, tr_order]
        if (not np.isfinite(xcoo)) or (xcoo <= 0.0):
            continue

        x1 = int(np.ceil(xcoo - ap1))
        x2 = int(np.floor(xcoo + ap1))

        if x1 < 0:
            x1 = 0
        if x2 >= nrows:
            x2 = nrows - 1
        if x2 < x1:
            continue

        flux_sum = 0.0
        for j in range(x1, x2 + 1):
            flux_sum += image[j, i]

        if skycorr:
            nap = x2 - x1 + 1
            flux_sum -= bkg_model[i] * nap
            flux_sum -= contam_model[i, tr_order]

        traceflux[i] = flux_sum

    return traceflux


@jit(nopython=True)
def _median_abs_dev_sigma(pts):
    med = np.median(pts)
    abs_dev = np.abs(pts - med)
    mad = np.median(abs_dev)
    sigma = 1.4826 * mad
    if sigma <= 0.0 and len(pts) > 1:
        sigma = stdev2(pts, med)
    return med, sigma


@jit(nopython=True)
def corrdata(tspec_med, tspec_med_t, sigcut=3.0):
    n1 = tspec_med.shape[0]
    n2 = tspec_med.shape[1]

    badmap = np.zeros((n1, n2), dtype=np.uint8)
    tspec_med_in = np.copy(tspec_med)
    tspec_med_t_in = np.copy(tspec_med_t)

    for j in range(n2):
        mean, std = _median_abs_dev_sigma(tspec_med_t_in[:, j])
        if std <= 0.0:
            continue

        for i in range(n1):
            if np.abs(tspec_med_t_in[i, j] - mean) > sigcut * std:
                badmap[i, j] = 1

    max_neighbors = 4
    pixels = np.zeros(max_neighbors)
    pixels_t = np.zeros(max_neighbors)

    for i in range(n1):
        for j in range(n2):
            if badmap[i, j] > 0:
                k = 0
                radius = 1
                while radius < n1 and k < max_neighbors:
                    i1 = i - radius
                    if i1 >= 0 and badmap[i1, j] == 0:
                        pixels[k] = tspec_med_in[i1, j]
                        pixels_t[k] = tspec_med_t_in[i1, j]
                        k += 1
                        if k >= max_neighbors:
                            break

                    i2 = i + radius
                    if i2 < n1 and badmap[i2, j] == 0:
                        pixels[k] = tspec_med_in[i2, j]
                        pixels_t[k] = tspec_med_t_in[i2, j]
                        k += 1

                    radius += 1

                if k > 0:
                    tspec_med[i, j] = np.median(pixels[:k])
                    tspec_med_t[i, j] = np.median(pixels_t[:k])
                else:
                    tspec_med[i, j] = tspec_med_in[i, j]
                    tspec_med_t[i, j] = tspec_med_t_in[i, j]

    return tspec_med, tspec_med_t
