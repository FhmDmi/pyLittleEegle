# Database module for EEG database management in NY format.
# v 0.2 Aug 2026
# Part of the Eegle package - Python version
# Copyright Fahim Doumi, CeSMA, Marco Congedo, CNRS, University Grenoble Alpes.
#
# ? ¤ CONTENT ¤ ? 
# This script comprised utils used in different modules

import os, numpy as np
from typing import List, Optional

def getFilesInDir(directory, ext=None, isin=""):
    """
    Get all files in a directory with optional filtering.
    
    Args:
        directory: path to directory
        ext: tuple of extensions to filter (e.g., ('.npz', '.yml'))
        isin: only include files containing this string in their name
    
    Returns:
        List of full paths to files
    """
    files = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path):
            # Filter by extension if provided
            if ext is not None:
                if not any(full_path.endswith(e) for e in ext):
                    continue
            # Filter by substring if provided
            if isin and isin not in item:
                continue
            files.append(full_path)
    return files


def getFoldersInDir(directory):
    """Get all subdirectories in a directory."""
    folders = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path):
            folders.append(full_path)
    return folders

def stim2mark(stim: List[int], 
              wl: int, 
              offset: int = 0, 
              code: Optional[List[int]] = None) -> List[List[int]]:
    """
    Convert a stimulation vector into marker vectors.
    
    Args:
        stim: the stimulation vector to be converted
        wl: the window (trial or ERP) length in samples.
    
    Optional Keyword Arguments:
        offset: offset for marker positions (default: 0)
        code: by default, the output will hold as many marker vectors as the largest tag 
              (integers) in stim, which may or may not hold instances of all integers up to the largest.
              If there are missing integers, the corresponding marker vector will be empty.
              Alternatively, a list of tags coding the classes of stimulations in stim can be passed as 
              kwarg code. In this case, arbitrary non-zero tags can be used (even negative)
              and the number of marker vectors will be equal to the number of
              unique integers in code. If code is provided, the marker vectors are arranged in the order given there,
              otherwise the first vector corresponds to the tag 1, the second to tag 2, etc.
              In any case, in each vector, the samples are sorted in ascending order.
    
    Warning:
        Markers which value plus the offset is non-positive or exceeds the length of stim minus wl 
        will be ignored, as they cannot define a complete ERP (or trial). If this happens, passing the 
        output to mark2stim will not return stim back exactly. Actually, calling this function and 
        reverting the operation with mark2stim ensures that the stimulation vector is valid.
    
    Returns:
        A list of z marker vectors, where z is the number of classes, i.e.,
        the highest integer in stim or the number of non-zero elements in code if it is provided.
    
    Examples:
        sr, wl = 128, 256  # sampling rate, window length of trials
        ns = sr * 100  # number of samples of the recording
        
        # simulate a valid stimulations vector for three classes
        import random
        stim = [random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]) for i in range(ns-wl)] + [0] * wl
        
        mark = stim2mark(stim, wl)
        stim2 = mark2stim(mark, ns)  # is identical to stim
    """
    if code is None:
        # Get unique non-zero values and create range from 1 to max
        unique_vals = set(stim) - {0}  # Remove 0
        if unique_vals:
            unic = list(range(1, max(unique_vals) + 1))
        else:
            unic = []
    else:
        unic = sorted(code)
    
    # Create marker vectors
    marker_vectors = []
    for j in unic:
        markers = [i + offset for i in range(len(stim)) 
                  if stim[i] == j and i + offset + wl - 1 <= len(stim) and i + offset > 0]
        marker_vectors.append(markers)
    
    return marker_vectors


def mark2stim(mark: List[List[int]], 
              ns: int, 
              offset: int = 0, 
              code: Optional[List[int]] = None) -> List[int]:
    """
    Reverse transformation of stim2mark.
    
    Args:
        mark: list of marker vectors
        ns: number of samples for the output stimulation vector
        offset: offset for marker positions (default: 0)
        code: optional code vector. If an offset has been used in stim2mark, 
              -offset must be used here in order to get back the original stimulation vector.
    
    Note:
        If code is provided, it must not contain 0.
    
    Returns:
        A stimulation vector of length ns
    
    Examples: see stim2mark
    """
    stim = [0] * ns
    
    if code is None:
        unic = [0] + list(range(1, len(mark) + 1))  # [0, 1, 2, ..., len(mark)]
    else:
        unic = [0] + sorted(code)
    
    for z in range(len(mark)):
        for j in mark[z]:
            if 0 <= j + offset < ns:  # Bounds checking
                stim[j + offset] = unic[z + 1]
    
    return stim


def car(X, correction=0, inplace=True):
    """
    Re-reference `X` to the Common Average Reference (CAR),
    setting the mean of the rows of `X` to zero (or adjusted by correction).
    Port of Eegle.jl's `car!`.

    Parameters
    ----------
    X : ndarray, shape (T, N)
        The EEG recording (T samples, N channels).
    correction : int or float, default=0
        Non-negative adjustment factor.
        - When 0: standard CAR (sum of rows divided by N).
        - When > 0: sum of rows divided by (N + correction).
        A value of 1 yields reference method 'B' from Kim2023GhostICs.
    inplace : bool, default=True
        If True, modifies `X` in-place (matching Julia `car!` behavior).
        If False, operates on a copy and leaves original `X` unchanged.

    Returns
    -------
    ndarray, shape (T, N)
        The re-referenced matrix.
    """
    if correction < 0:
        raise ValueError("pyLittleEegle `car`: `correction` must be non-negative.")

    # Ensure working on floats to prevent UFuncTypeError on in-place subtraction
    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float64)
        inplace = False  # Array was already copied during type conversion

    target = X if inplace else X.copy()
    n_channels = target.shape[1]

    denom = n_channels if correction == 0 else (n_channels + correction)
    avg = np.sum(target, axis=1, keepdims=True) / denom

    target -= avg
    return target


def global_field_power(X, func=None):
    """
    Compute the Global Field Power (GFP), defined as the sample-by-sample
    total EEG power (sum of squared channel potentials per time sample).
    Port of Eegle.jl's `globalFieldPower`.

    Parameters
    ----------
    X : ndarray, shape (T, N)
        The EEG recording (T samples, N channels).
    func : callable, optional
        Function applied element-wise to the output.

    Returns
    -------
    ndarray, shape (T,)
        GFP values for each sample.
    """
    gfp = np.sum(X**2, axis=1)
    return func(gfp) if func is not None else gfp


def global_field_rms(X, func=None):
    """
    Compute the Global Field Root Mean Square (GFRMS), defined as the square root
    of the GFP divided by the number of electrodes.
    Port of Eegle.jl's `globalFieldRMS`.

    Parameters
    ----------
    X : ndarray, shape (T, N)
        The EEG recording (T samples, N channels).
    func : callable, optional
        Function applied element-wise to the output (e.g., `np.log`).

    Returns
    -------
    ndarray, shape (T,)
        GFRMS values for each sample.
    """
    gfrms = np.sqrt(np.mean(X**2, axis=1))

    if func is not None:
        # Ignore divide-by-zero warnings to match Julia's silent -Inf handling on log(0)
        with np.errstate(divide="ignore"):
            return func(gfrms)
    return gfrms


def reject(X, stim, wl, offset=0, upper_limit=1.2, return_details=False):
    """
    Automatic rejection of artifacted trials in tagged EEG data
    via adaptive log-GFRMS amplitude thresholding.
    Port of Eegle.jl's `reject`.

    Parameters
    ----------
    X : ndarray, shape (T, N)
        The whole EEG recording (T samples, N channels).
    stim : array-like, shape (T,)
        Stimulation vector containing integer tags (0 indicates no event).
    wl : int
        Trial / ERP window length in samples.
    offset : int, default=0
        Offset in samples for marker positions passed to `stim2mark`.
    upper_limit : float, default=1.2
        Multiplier modulating the upper rejection threshold (typically in [1.0, 1.6]).
    return_details : bool, default=False
        If True, returns a 9-tuple containing debug thresholds and log-GFRMS array.

    Returns
    -------
    cleanstim : ndarray, shape (T,)
        Stimulation vector with rejected trials set to 0.
    rejecstim : ndarray, shape (T,)
        Stimulation vector containing only rejected trials (accepted set to 0).
    cleanmark : list of list of int
        Marker vectors for accepted trials, produced by `stim2mark`.
    rejecmark : list of list of int
        Marker vectors for rejected trials, produced by `stim2mark`.
    rejected : ndarray of int, shape (n_classes,)
        Count of rejected trials per unique class.
    (Optional details) : frms, m, thr_down, thr_up
    """
    ns, ne = X.shape
    stim = np.asarray(stim, dtype=int)

    if len(stim) != ns:
        raise ValueError(
            f"pyLittleEegle `reject`: `stim` length ({len(stim)}) does not match sample count in `X` ({ns})."
        )

    # Compute natural logarithm of GFRMS
    frms = global_field_rms(X, func=np.log)

    cleanstim = stim.copy()
    unique_tags = np.unique(stim)
    classcode = np.sort(unique_tags[unique_tags > 0])
    nc = len(classcode)
    rejected = np.zeros(nc, dtype=int)

    # Sorted log-GFRMS for adaptive thresholding
    p = np.argsort(frms)

    # Central tendency estimator m: mean of 2*wl values around median
    mid = ns // 2
    start_idx = max(0, mid - wl)
    end_idx = min(ns, mid + wl + 1)  # +1 to match Julia inclusive range
    m = np.mean(frms[p][start_idx:end_idx])

    # Lower threshold: 10th smallest value (index 9) to avoid outlier log near zero
    thr_down = frms[p][min(9, ns - 1)]

    # Upper threshold
    thr_up = m + ((m - thr_down) * upper_limit)

    stim_to_index = {val: i for i, val in enumerate(classcode)}

    # Pass 1: Reject epochs with near-zero signal (mean GFRMS < thr_down)
    skip_until = 0
    for s in range(ns - wl + 1):
        if s < skip_until:
            continue
        current_stim = cleanstim[s]
        if current_stim > 0:
            if np.mean(frms[s : s + wl]) < thr_down:
                skip_until = s + wl
                rejected[stim_to_index[current_stim]] += 1
                cleanstim[s : s + wl] = 0

    # Pass 2: Reject epochs with high-amplitude artifacts (max GFRMS > thr_up)
    skip_until = 0
    for s in range(ns - wl + 1):
        if s < skip_until:
            continue
        current_stim = cleanstim[s]
        if current_stim > 0:
            if np.max(frms[s : s + wl]) > thr_up:
                skip_until = s + wl
                rejected[stim_to_index[current_stim]] += 1
                cleanstim[s : s + wl] = 0

    rejecstim = stim - cleanstim

    # Generate marker vectors using internal stim2mark
    code_list = classcode.tolist()
    cleanmark = stim2mark(cleanstim.tolist(), wl, offset=offset, code=code_list)
    rejecmark = stim2mark(rejecstim.tolist(), wl, offset=offset, code=code_list)

    if return_details:
        return cleanstim, rejecstim, cleanmark, rejecmark, rejected, frms, m, thr_down, thr_up
    return cleanstim, rejecstim, cleanmark, rejecmark, rejected