# Database module for EEG database management in NY format.
# v 0.1 Dec 2025
# Part of the Eegle package - Python version
# Copyright Fahim Doumi, CeSMA, Marco Congedo, CNRS, University Grenoble Alpes.
#
# ? ¤ CONTENT ¤ ? 
#
# STRUCTURES
# EEG | holds data and metadata of an EEG recording
#
# FUNCTIONS:
# readNY        | read an EEG recording in [NY format](@ref)

import numpy as np
import yaml
import os 
import warnings
import utils
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple
from scipy import signal


@dataclass(frozen=True)
class EEG:
    """
    Data structure for an EEG BCI (Brain-Computer Interface) session, holding data and metadata.
    
    Python translation of the Julia EEG structure from Eegle.jl package.
    While conceived specifically for BCI sessions, the structure can be used also for general EEG recordings.
    
    Fields:
        id: Dictionary of the .yml file metadata (run, timestamp, database, subject, session, condition, paradigm)
        acquisition: Dictionary of acquisition metadata (sensors, software, ground, reference, filter, sensortype, samplingrate, hardware)
        documentation: Dictionary of documentation metadata (doi, repository, description, investigators, place)
        formatversion: Format version field of the .yml file
        
        # Most useful fields in practice:
        db: Name of the database to which the recording belongs
        paradigm: BCI paradigm (e.g., 'MI', 'P300', 'ERP')
        subject: Serial number of the present subject in the above database
        session: Serial number of the present session for the above subject
        run: Serial number of the present run of the above session
        sensors: Labels of the scalp electrode leads in standard notation (10-20, 10-10,...)
        sr: Sampling rate in samples per second
        ne: Number of electrode leads
        ns: Number of samples
        wl: Window length in samples. Typically, the duration of a BCI trial
        offset: Trial offset (see offset documentation)
        nc: Number of classes (non-zero tags)
        clabels: Labels of the classes
        stim: The stimulation vector
        mark: The marker vectors (list of lists of sample indices for each class)
        y: The non-zero tags of stim as a vector. Each tag is the class label of the corresponding trial
        X: The T×N EEG data array, with T and N the number of samples and channels (sensors), respectively
        trials: A list of trials, each of size N×wl, extracted in the order of tags given in stim (optional)
    """
    
    # Metadata dictionaries
    id: Dict[Any, Any]
    acquisition: Dict[Any, Any] 
    documentation: Dict[Any, Any]
    formatversion: str
    
    # Core fields
    dbName: str
    condition: str
    paradigm: str  
    subject: int
    session: int
    run: int
    sensors: List[str]
    sr: int  # sampling rate
    ne: int  # number of electrodes
    ns: int  # number of samples
    wl: int  # window length
    offset: int
    nc: int  # number of classes
    clabels: List[str]  # class labels
    stim: List[int]  # stimulation vector
    mark: List[List[int]]  # marker vectors
    y: List[int]  # class labels for each trial
    X: np.ndarray  # EEG data matrix (ns × ne)
    trials: Optional[List[np.ndarray]]  # individual trials (optional)
    
    def __post_init__(self):
        """Validation after initialization"""
        # Basic consistency checks
        if self.X.shape[0] != self.ns:
            raise ValueError(f"X samples ({self.X.shape[0]}) doesn't match ns ({self.ns})")
        if self.X.shape[1] != self.ne:
            raise ValueError(f"X channels ({self.X.shape[1]}) doesn't match ne ({self.ne})")
        if len(self.sensors) != self.ne:
            raise ValueError(f"Number of sensor labels ({len(self.sensors)}) doesn't match ne ({self.ne})")
        if len(self.stim) != self.ns:
            raise ValueError(f"Stimulation vector length ({len(self.stim)}) doesn't match ns ({self.ns})")
        if len(self.mark) != self.nc:
            raise ValueError(f"Number of marker vectors ({len(self.mark)}) doesn't match nc ({self.nc})")
        if len(self.clabels) != self.nc:
            raise ValueError(f"Number of class labels ({len(self.clabels)}) doesn't match nc ({self.nc})")
    

    
    def __repr__(self) -> str:
        """Custom representation similar to Julia's show function"""
        r, c = self.X.shape
        dtype = self.X.dtype
        
        return f"""∿ EEG Data type; {r} x {c}
∼∽∿∽∽∽∿∼∿∽∿∽∿∿∿∼∼∽∿∼∽∽∿∼∽∽∼∿∼∿∿∽∿∽∼∽∽∿∽∽
NY format version (.formatversion): {self.formatversion}
∼∽∿∽∽∽∿∼∿∽∿∽∿∿∿∼∼∽∿∼∽∽∿∼∽∽∼∿∼∿∿∽∿∽∼∽∽∿∽∽
.dbName            : {self.dbName}
.condition         : {self.condition}
.paradigm          : {self.paradigm}
.subject           : {self.subject}
.session           : {self.session}
.run               : {self.run}
.sensors           : {len(self.sensors)}-List[str]
.sr(samp. rate)    : {self.sr}
.ne(# electrodes)  : {self.ne}
.ns(# samples)     : {self.ns}
.wl(win. length)   : {self.wl}
.offset            : {self.offset}
.nc(# classes)     : {self.nc}
.clabels(c=class)  : {len(self.clabels)}-List[str]
.stim(ulations)    : {len(self.stim)}-List[int]
.mark(ers)         : {[len(self.mark[i]) for i in range(len(self.mark))]}-Lists[int]
.y (all c labels)  : {len(self.y)}-List[int]
.X (EEG data)      : {r}x{c}-ndarray[{dtype}]
.trials            : {'None' if self.trials is None else f'{len(self.trials)}-List[ndarray[{self.trials[0].dtype if self.trials else "unknown"}]]'}
Dict: .id, .acquisition, .documentation"""
    

def _standardizeClasses(paradigm: str, 
                        clabels: List[str], 
                        clabelsval: List[int], 
                        stim: List[int]) -> Tuple[List[int], List[str]]:
    """
    `_standardizeClasses` function is exclusively used within `readNY` to normalize 
    EEG data numerical codes according to standard conventions.
    
    It takes an experimental paradigm (MI, P300, ERP), class names (`clabels`), 
    the related numerical values(`clabelsval`), and the stim vector, then applies 
    a uniform mapping (e.g., "left_hand" → 1, "right_hand" → 2 for MI). 
    
    The function verifies class compatibility with the chosen paradigm, detects if 
    data is already standardized, and returns a normalized stim vector, thereby 
    facilitating model training across heterogeneous databases.
    
    This function is case insensitive but you need to respect the correct spelling of classes.
    - MI: left_hand, right_hand, feet, rest, both_hands, tongue
    - P300: nontarget, target
    - ERP: not currently supported.
    
    Returns a new standardized stim vector and clabels if it was not already the same mapping.
    """
    
    # Define standardized mappings for each paradigm
    if paradigm.upper() == "MI":
        standard_mapping = {
            "left_hand": 1, "right_hand": 2, "feet": 3, 
            "rest": 4, "both_hands": 5, "tongue": 6
        }
        supported_classes = list(standard_mapping.keys())
    elif paradigm.upper() == "P300":
        standard_mapping = {"nontarget": 1, "target": 2}
        supported_classes = list(standard_mapping.keys())
    elif paradigm.upper() == "ERP":
        raise ValueError(
            "utils.py package, internal function `_standardizeClasses` called by `readNY`: "
            "ERP paradigm not supported yet for class standardization"
        )
    else:
        raise ValueError(
            f"utils.py package, internal function `_standardizeClasses` called by `readNY`: "
            f"Unknown paradigm: {paradigm}. Supported paradigms: MI, P300"
        )
    
    # Check for unsupported classes (case insensitive)
    clabels_lower = [label.lower() for label in clabels]
    unsupported_classes = [label for label in clabels_lower if label not in standard_mapping]
    
    # Throw error if unsupported classes found
    if unsupported_classes:
        error_msg = (
            f"utils.py package, internal function `_standardizeClasses` called by `readNY`: "
            f"only these classes are compatible with standardization for {paradigm} paradigm: "
            f"{', '.join(supported_classes)}. "
            f"\nUnsupported classes found: {', '.join(unsupported_classes)}. "
            f"\nPlease verify the correct spelling of your classes (case insensitive)"
        )
        raise ValueError(error_msg)
    
    # Create mapping and check if already standardized
    value_mapping = {clabelsval[i]: standard_mapping[clabels_lower[i]] 
                    for i in range(len(clabels_lower))}
    already_standardized = all(k == v for k, v in value_mapping.items())
    
    stim_standardized = stim.copy()
    clabels_standardized = clabels.copy()
    
    if already_standardized:
        print("✓ Class labels in file follow Eegle's conventions.")
        pass
    else:
        # Apply standardization mapping to stim vector
        for i in range(len(stim_standardized)):
            if stim_standardized[i] != 0 and stim_standardized[i] in value_mapping:
                stim_standardized[i] = value_mapping[stim_standardized[i]]
        
        # Reorder clabels according to standardized mapping
        sorted_indices = sorted(range(len(clabels_lower)), 
                              key=lambda i: standard_mapping[clabels_lower[i]])
        clabels_standardized = [clabels[i] for i in sorted_indices]
        
        # Create mapping display for user feedback
        mapping_display = []
        for k, v in value_mapping.items():
            original_label = clabels[clabelsval.index(k)]
            mapping_display.append(f"{original_label}({k}->{v})")
        
        print(f"\n✓ Class labels have been formatted according to Eegle's convention")
        print(f"Mapping applied: {', '.join(mapping_display)}")
    
    return stim_standardized, clabels_standardized


def readNY(filename: str,
           toFloat64: bool = True,
           bandStop: Tuple = (),
           bandPass: Tuple = (),
           bsDesign: int = 8,
           bpDesign: int = 4,
           classes: Union[bool, List[str]] = True,
           stdClass: bool = True,
           msg: str = "") -> EEG:
    """
    Read EEG/BCI data in NY format, preprocess them if desired, and create an EEG structure.
    
    If requested, the preprocessing operations are performed in the order of the kwargs
    as listed here below.
    
    Args:
        filename: the complete path of either the *.npz* or the *.yml* file of the recording to be read.
    
    Optional Keyword Arguments:
        toFloat64: if true, the EEG data is converted to Float64 if it is not already (default: True)
        bandStop: a 2-tuple holding the limits in Hz of a notch filter (default: no filter)
        bandPass: a 2-tuple holding the limits in Hz of a band-pass filter (default: no filter)
        bsDesign: the order of filter design method for the notch filter (default: 8)
        bpDesign: the order of filter design method for the band-pass filter (default: 4)
        classes: 
            - if True (default), the .trials field of the EEG structure is filled with the trials for all classes
            - If it is a list of class labels (for example, classes=["left_hand", "right_hand"]), 
              only the trials with those class labels will be stored. 
              The tags corresponding to each class labels will be replaced by natural numbers (1, 2,...)
              and written in the .stim and y fields of the output
            - If False, the field trials of the returned EEG structure will be set to None.
        stdClass: 
            - if True (default), a standardization is applied to the class labels, according to predefined 
              conventions to facilitate transfer learning and model training across heterogeneous databases.
              The standardization applies uniform numerical codes regardless of the original database encoding:
              - **MI paradigm**: "left_hand" → 1, "right_hand" → 2, "feet" → 3, "rest" → 4, "both_hands" → 5, "tongue" → 6
              - **P300 paradigm**: "nontarget" → 1, "target" → 2
              - **ERP paradigm**: not currently supported
            - if False, original class labels and their corresponding numerical values are preserved as found in the database
            The standardization is case-insensitive, but requires correct spelling of class names.
            When used with classes as a list of class labels, standardization is applied after class selection.
            If class labels are already standardized, the original mapping is preserved.
            It is recommended to leave the default setting for stdClass (True) when all relevant classes 
            are available in your database configuration.
        msg: print string msg on exit if it is not empty. By default it is empty.
    
    Notes:
        - If you use resampling, the new sampling rate will be rounded to the nearest integer.
        - If the field offset of the NY file is different from zero,
          the stimulations in stim and markers in mark will be shifted to account for the offset.
          The field .offset will then be reset to zero.
    
    Returns:
        An EEG data structure.
    
    Examples:
        # Using examples data provided by Eegle
        o = readNY(EXAMPLE_P300_1)
        
        # filter the data and do artifact-rejection by adaptive amplitude thresholding
        o = readNY(EXAMPLE_P300_1, bandPass=(1, 24))
        
        # read the whole recording, but store in o.trials the trials 
        # only for classes "right_hand" and "feet" (exclude "rest")
        o = readNY(EXAMPLE_MI_1,
                   bandPass=(1, 24), 
                   classes=["right_hand", "feet"])
    """
    # Read data file (.npz) and info file (.yml)
    base_filename = os.path.splitext(filename)[0]
    
    # Load NPZ data file
    data = np.load(base_filename + ".npz")
    
    # Load YAML info file
    with open(base_filename + ".yml", 'r') as f:
        info = yaml.safe_load(f)
    
    # Extract sampling rate
    sr = info["acquisition"]["samplingrate"]
    if sr != round(sr) or not isinstance(sr, (int, float)):
        warnings.warn("readNY: the sampling rate is not an integer. It will be rounded to the nearest integer")
        sr = round(sr)
    sr = int(sr)
    
    # Extract stimulations
    stim = data["stim"].tolist()  # Convert numpy array to list for consistency with EEG structure
    
    # Extract paradigm (June 2025, added forC)
    paradigm = info["id"]["paradigm"]  # Using string instead of Symbol like in Julia
    
    # Extract number of samples and electrodes
    ns, ne = data["data"].shape
    
    # Extract offset for trial starting sample
    offset = info["stim"]["offset"]
    if offset != round(offset) or not isinstance(offset, (int, float)):
        warnings.warn("readNY: the offset is not an integer. It will be rounded to the nearest integer")
        offset = round(offset)
    offset = int(offset)
    
    # Extract trial duration (window length)
    wl = info["stim"]["windowlength"]
    if wl != round(wl) or not isinstance(wl, (int, float)):
        warnings.warn("readNY: the trial duration (windowlength) is not an integer. It will be rounded to the nearest integer")
        wl = round(wl)
    wl = int(wl)
    
    # Convert to Float64 and apply filtering if requested
    X = None
    conversion = data["data"].dtype != np.float64 and toFloat64

    # Apply bandstop filter if requested
    if bandStop:
        nyquist = sr / 2
        low = bandStop[0] / nyquist
        high = bandStop[1] / nyquist
        b, a = signal.butter(bsDesign, [low, high], btype='bandstop')
        padlen = min(3 * (max(len(b), len(a)) - 1), len(input_data) - 1) 
        X = signal.filtfilt(b, a, data["data"].astype(np.float64) if conversion else data["data"], 
                        padtype='odd', padlen=padlen, method='pad', axis=0)

    if bandPass:
        nyquist = sr / 2
        low = bandPass[0] / nyquist
        high = bandPass[1] / nyquist
        b, a = signal.butter(bpDesign, [low, high], btype='bandpass')
        input_data = X if X is not None else (data["data"].astype(np.float64) if conversion else data["data"])
        padlen = min(3 * (max(len(b), len(a)) - 1), input_data.shape[0] - 1)

        X = signal.filtfilt(b, a, input_data, padtype='odd', padlen=padlen, method='pad', axis=0)
    
    # If no filtering applied, just handle conversion
    if not bandStop and not bandPass:
        X = data["data"].astype(np.float64) if conversion else data["data"]

    # Added April-June 2025 to allow loading a file keeping only the chosen classes
    stim = [int(x) for x in stim]  # Convert to list of integers (equivalent to Vector{Int64})
    nc = info["stim"]["nclasses"]  # Number of classes
    labels_dict = sorted(info["stim"]["labels"].items(), key=lambda x: x[1])  # Sort by value
    clabels = [pair[0] for pair in labels_dict]  # Class labels (keys)
    clabelsval = [pair[1] for pair in labels_dict]  # Class values (values)

    if isinstance(classes, list):  # June 2025: classes is now Union[bool, List[str]]
        missing_classes = set(classes) - set(clabels)
        if missing_classes:
            error_msg = (f"utils.py, function `readNY`: classes not found: "
                        f"{', '.join(missing_classes)}. Available classes: {', '.join(clabels)}")
            raise ValueError(error_msg)

        classes_val = []  # memory efficient to declare type
        classes_val = [clabelsval[clabels.index(c)] for c in classes]  # get stim values corresponding to classes selected

        un = sorted(set(stim))[1:]  # unique non-zero values, sorted (exclude 0)
        if not (set(un) & set(classes_val)):  # intersection is empty
            raise ValueError("utils.py, function `readNY`: the stimulations do not contain the classes requested in classes")
        
        elimina = set(un) - set(classes_val)  # values to eliminate
        if elimina:
            for i in range(len(stim)):
                if stim[i] in elimina:
                    stim[i] = 0

        nc = len(classes)
        clabels = [c for c in clabels if c in classes]
        clabelsval = [c for c in clabelsval if c in classes_val]  # needed for stdClass

        if stdClass:  # STANDARDIZE classes if std_class is set to True and classes is a list
            stim, clabels = _standardizeClasses(paradigm, clabels, clabelsval, stim)
    else:
        if stdClass:  # STANDARDIZE classes if std_class is set to True and classes is a bool
            stim, clabels = _standardizeClasses(paradigm, clabels, clabelsval, stim)

    # Make sure all stimulations+offset+wl does not exceed the recording duration
    for s in range(len(stim)):
        if stim[s] > 0 and s + offset + wl - 1 > X.shape[0]:
            stim[s] = 0
            warnings.warn(f"utils.py, function `readNY`: the {s}th stimulation at sample {stim[s]} "
                        f"with offset {offset} and trial duration {wl} defines a trial exceeding the "
                        f"recording length. The stimulation has been eliminated.")


    # Convert stim to markers with offset, then back to stim (applies offset)
    mark = utils.stim2mark(stim, wl, offset=offset, code=sorted(set(stim))[1:])  # exclude 0
    stim = utils.mark2stim(mark, ns)  # new stim with offset taken into account

    if offset != 0:  # offset reset to 0
        print(f"✓ Initial offset ({offset} samples) has been applied and `offset` has been reset to 0")
        offset = 0

    # Verify number of classes matches number of markers
    if len(mark) != nc:
        raise RuntimeError("utils.py, function `readNY`: the number of classes in .mark does not correspond to the number of markers found in .stim")

    # Extract trials if classes is not False
    trials = None if classes is False else [
        X[mark[i][j]:mark[i][j] + wl, :]
        for i in range(nc)
        for j in range(len(mark[i]))
    ]

    if msg:  # if msg is not empty
        print(msg)

    # Create the EEG structure
    return EEG(
        id=info["id"],
        acquisition=info["acquisition"],
        documentation=info["documentation"],
        formatversion=info["formatversion"],
        dbName=info["id"]["database"],
        condition=info["id"]["condition"],
        paradigm=paradigm,
        subject=info["id"]["subject"],
        session=info["id"]["session"],
        run=info["id"]["run"],
        sensors=info["acquisition"]["sensors"],
        sr=sr,
        ne=ne,
        ns=ns,
        wl=wl,
        offset=offset,  # trials offset
        nc=nc,
        clabels=clabels,
        stim=stim,
        mark=mark,
        y=[i+1 for i in range(nc) for j in range(len(mark[i]))],  # y: all labels
        X=X,  # whole EEG recording
        trials=trials  # all trials, by class, if requested, None otherwise
    )

