# Database module for EEG database management in NY format.
# v 0.1 Dec 2025
# Part of the Eegle package - Python version
# Copyright Fahim Doumi, CeSMA, Marco Congedo, CNRS, University Grenoble Alpes.
#
# ? ¤ CONTENT ¤ ? 
# This script comprised utils used in different modules

import os
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

