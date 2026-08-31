# Database module for EEG database management in NY format.
# v 0.2 Aug 2026
# Part of the Eegle package - Python version
# Copyright Fahim Doumi, CeSMA, Marco Congedo, CNRS, University Grenoble Alpes.
#
# ? ¤ CONTENT ¤ ? 
#
# - InfoDB: dataclass holding information summarizing an EEG database
# - loadDB: return a list of .npz files in a directory
# - infoDB: print and return information about a database
# - selectDB: select database folders based on paradigm and class requirements

import os
import warnings
from dataclasses import dataclass
import yaml
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable


# Import your already converted functions
from .utils import getFilesInDir, getFoldersInDir

@dataclass
class InfoDB:
    """
    Immutable dataclass holding the summary information and metadata 
    of an EEG database (DB) in NY format.
    
    It is created by functions infoDB() and selectDB().
    
    Attributes:
        dbName: name or identifier of the database
        condition: experimental condition under which the DB has been recorded
        paradigm: for BCI data, this may be 'P300', 'ERP' or 'MI'
        files: list of .npz files, each corresponding to a session
        nSessions: list holding the number of sessions per subject
        nTrials: dict mapping each class label to a list of trials per session
        nSubjects: total number of subjects composing the DB
        nSensors: number of sensors (e.g., EEG electrodes)
        sensors: list of sensor labels (e.g., ['Fz', 'Cz', 'Oz'])
        sensorType: type of sensors (wet, dry, Ag/Cl, ...)
        nClasses: number of classes for which labels are available
        cLabels: list of class labels
        sr: sampling rate of the recordings (in samples)
        wl: for BCI, duration of trials (in samples)
        offset: shift to be applied to markers (in samples)
        filter: temporal filter applied to the data
        doi: digital object identifier (DOI) of the database
        hardware: equipment used (typically, the EEG amplifier)
        software: software used to obtain the recordings
        reference: label of the reference electrode
        ground: label of the electrical ground electrode
        place: place where recordings were obtained
        investigators: investigator(s) who obtained the recordings
        repository: public repository where the DB is accessible
        description: general description of the DB
        timestamp: date of publication of the DB
        formatVersion: version of the NY format
    """
    dbName: str
    condition: str
    paradigm: str
    files: List[str]
    nSessions: List[int]
    nTrials: Dict[str, List[int]]
    nSubjects: int
    nSensors: int
    sensors: List[str]
    sensorType: str
    nClasses: int
    cLabels: List[str]
    sr: int
    wl: int
    offset: int
    filter: str
    hardware: str
    software: str
    reference: str
    ground: str
    doi: str
    place: str
    investigators: str
    repository: str
    description: str
    timestamp: int
    formatVersion: str
    
    def __repr__(self) -> str:
        """Custom representation similar to Julia's show function"""
        import math
        
        # Format ntrialsperclass - show mean ± std + min,max
        trials_parts = []
        for class_name in self.cLabels:  # use cLabels to maintain order
            trials_vec = self.nTrials[class_name]
            if len(set(trials_vec)) == 1:  # All trials are the same for this class
                trial_str = f"{trials_vec[0]} ± 0"
                minmax_str = ""
            else:  # Calculate mean, std, min, max
                mean_trials = round(sum(trials_vec) / len(trials_vec), 1)
                variance = sum((x - mean_trials)**2 for x in trials_vec) / (len(trials_vec) - 1)
                std_trials = round(math.sqrt(variance), 1)
                min_trials = min(trials_vec)
                max_trials = max(trials_vec)
                trial_str = f"{mean_trials} ± {std_trials}"
                minmax_str = f"({min_trials},{max_trials})"
            trials_parts.append(f"{class_name}: {trial_str} {minmax_str}")
        
        # Format the display with proper spacing
        first_line = f"nTrials per class              : {trials_parts[0]}"
        if len(trials_parts) > 1:
            remaining_classes = "\n                                 ".join(trials_parts[1:])
            second_line = f"└▶mean ± std (min,max)           {remaining_classes}"
        else:
            second_line = "└▶mean ± std (min,max)"
        
        nTrials_str = f"{first_line}\n{second_line}"
        
        # Format sensors - show first 3 + total count if more than 3
        if len(self.sensors) <= 3:
            sensors_str = ", ".join(self.sensors)
        else:
            sensors_str = ", ".join(self.sensors[:3]) + "..."
        
        # Format nsessions - show single value if min == max
        min_sessions = min(self.nSessions)
        max_sessions = max(self.nSessions)
        nsessions_str = f"{min_sessions}" if min_sessions == max_sessions else f"({min_sessions},{max_sessions})"
        
        # Build the output string
        output = f"""🗄️  Database Summary: {self.dbName} | {self.nSubjects} subjects, {self.nClasses} classes
∼∽∿∽∽∽∿∼∿∽∿∽∿∿∿∼∼∽∿∼∽∽∿∼∽∽∼∿∼∿∿∽∿∽∼∽∼∿∼∿∿∽∿∽∼∽∼∽∽∼∿∼∿∿∽∿∼∿∿∽∿∼∿∿∽∿
NY format database main characteristics and metadata
∼∽∿∽∽∽∿∼∿∽∿∽∿∿∿∼∼∽∿∼∽∽∿∼∽∽∼∿∼∿∿∽∿∽∼∽∼∿∼∿∿∽∿∽∼∽∼∽∽∼∿∼∿∿∽∿∼∿∿∽∿∼∿∿∽∿
condition                      : {self.condition}
paradigm                       : {self.paradigm}
nSessions (min,max)            : {nsessions_str}
nSensors                       : {self.nSensors}
sensors                        : {sensors_str}
sensorType                     : {self.sensorType}
sr (Hz)                        : {self.sr}
wl (samples)                   : {self.wl}
offset (samples)               : {self.offset}
{nTrials_str}
∼∽∿∽∽∽∿∼∿∽∿∽∿∿∿∼∼∽∿∼∽∽∿∼∽∽∼∿∼∿∿∽∿∽∼∽∼∿∼∿∿∽∿∽∼∽∼∽∽∼∿∼∿∿∽∿∼∿∿∽∿∼∿∿∽∿
Fourteen Additional fields:
.files, .cLabels, .filter, .hardware, .software,
.doi, .reference, .ground, .place, .investigators,
.description, .repository, .timestamp, .formatVersion"""
        
        return output




def loadDB(corpusDir: str, isin: str = "") -> List[str]:
    """
    Return a list of the complete paths of all .npz files found in a directory.
    
    For each NPZ file, there must be a corresponding YAML metadata file with 
    the same name and extension .yml, otherwise the file is not included in the list.
    
    Args:
        corpusDir: directory path containing the database files
        isin: if provided, only files whose name contains this string are included
        
    Returns:
        List of complete paths to .npz files
        
    Examples:
        >>> files = loadDB("/path/to/database")
        >>> files = loadDB("/path/to/database", isin="subject01")
    """
    # Create a list of all .npz files found in corpusDir (complete path)
    npzFiles = getFilesInDir(corpusDir, ext=(".npz",), isin=isin)
    
    # Check if for each .npz file there is a corresponding .yml file
    missingYML = []
    for i, npz_file in enumerate(npzFiles):
        yml_file = os.path.splitext(npz_file)[0] + ".yml"
        if not os.path.isfile(yml_file):
            missingYML.append(i)
    
    if missingYML:
        warnings.warn("Database.loadDB: the following .yml files have not been found:")
        for i in missingYML:
            print(os.path.splitext(npzFiles[i])[0] + ".yml")
        
        # Remove files without corresponding .yml
        for i in reversed(missingYML):
            del npzFiles[i]
        print(f"\n{len(npzFiles)} files have been retained.")
    
    return npzFiles


def infoDB(corpusDir: str) -> InfoDB:
    """
    Create an InfoDB structure and show it in the console.
    
    The only argument (corpusDir) is the directory holding all files of a database
    in NY format.
    
    This function carries out sanity checks on the database and prints warnings
    if the checks fail.
    
    Args:
        corpusDir: directory path containing the database files
        
    Returns:
        InfoDB object containing database information
        
    Examples:
        >>> DB = infoDB("/path/to/database")
    """
    files = loadDB(corpusDir)
    
    # Make sure only .npz files have been passed
    files = [f for f in files if os.path.splitext(f)[1] == ".npz"]
    
    if len(files) == 0:
        raise ValueError("Database.infoDB: there are no .npz files in the list")
    
    # Read one YAML file to initialize lists
    filename = files[0]
    yml_file = os.path.splitext(filename)[0] + ".yml"
    if not os.path.isfile(yml_file):
        raise FileNotFoundError(f"Database.infoDB: no .yml file found for {filename}")
    
    with open(yml_file, 'r') as f:
        info = yaml.safe_load(f)
    
    # Initialize lists for all metadata fields
    sensors = []
    sensorType = []
    ground = []
    reference = []
    filter_list = []
    sr = []
    hardware = []
    software = []
    
    wl = []
    labels = []
    offset = []
    nClasses = []
    nTrials = []
    
    timestamp = []
    run = []
    condition = []
    dbName = []
    paradigm = []
    subject = []
    session = []
    
    place = []
    investigators = []
    doi = []
    repository = []
    description = []
    
    formatversion = []
    
    # Read all YAML files
    for filename in files:
        yml_file = os.path.splitext(filename)[0] + ".yml"
        if not os.path.isfile(yml_file):
            raise FileNotFoundError(f"Database.infoDB: no .yml file found for {filename}")
        
        with open(yml_file, 'r') as f:
            info = yaml.safe_load(f)
        
        acq = info["acquisition"]
        sensors.append(acq["sensors"])
        ground.append(acq["ground"])
        reference.append(acq["reference"])
        filter_list.append(acq["filter"])
        sensorType.append(acq["sensortype"])
        sr.append(acq["samplingrate"])
        hardware.append(acq["hardware"])
        software.append(acq["software"])
        
        stim = info["stim"]
        wl.append(stim["windowlength"])
        labels.append(stim["labels"])
        offset.append(stim["offset"])
        nClasses.append(stim["nclasses"])
        nTrials.append(stim["trialsperclass"])
        
        id_info = info["id"]
        timestamp.append(id_info["timestamp"])
        run.append(id_info["run"])
        condition.append(id_info["condition"])
        dbName.append(id_info["database"])
        paradigm.append(id_info["paradigm"])
        subject.append(id_info["subject"])
        session.append(id_info["session"])
        
        doc = info["documentation"]
        place.append(doc["place"])
        investigators.append(doc["investigators"])
        doi.append(doc["doi"])
        repository.append(doc["repository"])
        description.append(doc["description"])
        
        formatversion.append(info["formatversion"])
    
    # Warnings counter
    nwarnings = 0
    
    def mywarn(text: str):
        nonlocal nwarnings
        nwarnings += 1
        warnings.warn(f"Database.infoDB: {text}")
    
    # Helper function to compare lists/dicts for uniqueness
    def stringify(obj):
        """Convert object to string for comparison"""
        if isinstance(obj, (list, dict)):
            return str(sorted(obj.items()) if isinstance(obj, dict) else sorted(obj))
        return str(obj)
    
    # Check critical field consistency (warn if not unique)
    if len(set(paradigm)) > 1:
        mywarn("Paradigm is not unique across the database")
    if len(set(nClasses)) > 1:
        mywarn("Number of classes is not unique across the database")
    if len(set(stringify(l) for l in labels)) > 1:
        mywarn("Class labels are not unique across the database")
    if len(set(sr)) > 1:
        mywarn("Sampling rate is not unique across the database")
    if len(set(wl)) > 1:
        mywarn("Trial duration (windowlength) is not unique across the database")
    if len(set(offset)) > 1:
        mywarn("Trial offset is not unique across the database")
    
    # CRITICAL ERROR CHECK: unicity of triplets (subject, session, run)
    ssr = [(s, sess, r) for s, sess, r in zip(subject, session, run)]
    if len(set(ssr)) < len(subject):
        raise ValueError("Database.infoDB: there are duplicated triplets (subject, session, run)")
    
    # CRITICAL ERROR CHECK: session count consistency
    usub = list(set(subject))
    sess = [sum(1 for s in subject if s == sub) for sub in usub]  # sessions per subject
    if sum(sess) != len(files):
        raise ValueError("Database.infoDB: number of sessions doesn't match number of files")
    
    # Warning about run field inconsistency
    if len(set(run)) > 1:
        mywarn("field 'run' should be the same in all recordings")
    
    if nwarnings > 0:
        print(f"\n⚠ Be careful, {nwarnings} warnings have been found")
    
    # Extract main information (take first unique value)
    db_dbName = list(set(dbName))[0]
    db_condition = list(set(condition))[0]
    db_paradigm = list(set(paradigm))[0]
    db_files = files
    db_nSubjects = len(set(subject))
    db_nSessions = sess
    
    # Handle sensors - keep as list
    unique_sensors = []
    for s in sensors:
        s_str = stringify(s)
        if s_str not in [stringify(u) for u in unique_sensors]:
            unique_sensors.append(s)
    db_sensors = unique_sensors[0]
    db_nSensors = len(db_sensors)
    
    db_sensorType = list(set(sensorType))[0]
    db_nClasses = list(set(nClasses))[0]
    db_sr = list(set(sr))[0]
    db_wl = list(set(wl))[0]
    db_offset = list(set(offset))[0]
    db_filter = list(set(filter_list))[0]
    db_doi = list(set(doi))[0]
    db_hardware = list(set(hardware))[0]
    db_software = list(set(software))[0]
    db_reference = list(set(reference))[0]
    db_ground = list(set(ground))[0]
    db_place = list(set(place))[0]
    db_investigators = list(set(investigators))[0]
    db_repository = list(set(repository))[0]
    db_description = list(set(description))[0]
    db_timestamp = list(set(timestamp))[0]
    db_formatVersion = list(set(formatversion))[0]
    
    # Extract class labels in correct order (sorted by stim values)
    # labels[0] should be a dict like {"left_hand": 1, "right_hand": 2}
    all_labels = labels[0]
    if isinstance(all_labels, dict):
        sorted_labels = sorted(all_labels.items(), key=lambda x: x[1])
        db_cLabels = [label[0] for label in sorted_labels]
    else:
        db_cLabels = list(all_labels)
    
    # Extract trials per class per session
    db_nTrials = {}
    
    for class_name in db_cLabels:
        trials = []
        for trial_dict in nTrials:
            if class_name in trial_dict:
                trials.append(trial_dict[class_name])
            else:
                trials.append(0)  # no trials for this class in this session
        db_nTrials[class_name] = trials
    
    # Create and return infoDB structure
    return InfoDB(
        dbName=db_dbName,
        condition=db_condition,
        paradigm=db_paradigm,
        files=db_files,
        nSessions=db_nSessions,
        nTrials=db_nTrials,
        nSubjects=db_nSubjects,
        nSensors=db_nSensors,
        sensors=db_sensors,
        sensorType=db_sensorType,
        nClasses=db_nClasses,
        cLabels=db_cLabels,
        sr=db_sr,
        wl=db_wl,
        offset=db_offset,
        filter=db_filter,
        hardware=db_hardware,
        software=db_software,
        reference=db_reference,
        ground=db_ground,
        doi=db_doi,
        place=db_place,
        investigators=db_investigators,
        repository=db_repository,
        description=db_description,
        timestamp=db_timestamp,
        formatVersion=db_formatVersion
    )

class CaseInsensitiveList(list):
    """A list of strings that allows case-insensitive 'in' checks."""
    def __contains__(self, item):
        if isinstance(item, str):
            return item.lower() in (str(x).lower() for x in self)
        return super().__contains__(item)

def _get_nested_value(data: Dict, path: str) -> Any:
    """Extract value from nested dictionary using dot-separated path."""
    shortcuts = {
        "sr": "acquisition.samplingrate",
        "ref": "acquisition.reference",
        "tpc": "stim.trialsperclass",
        "perfLHRH": "perf.left_hand-right_hand",
        "perfRHF": "perf.right_hand-feet"
    }
    
    resolved_path = shortcuts.get(path, path)
    
    for shortcut, full_path in shortcuts.items():
        if path.startswith(shortcut + "."):
            resolved_path = path.replace(shortcut, full_path, 1)
            break
            
    keys_path = resolved_path.split('.')
    
    if len(keys_path) == 1:
        key = keys_path[0]
        if key in data:
            return data[key]
            
        # Recursive search
        def _find(d, k):
            if k in d: return d[k]
            for v in d.values():
                if isinstance(v, dict):
                    res = _find(v, k)
                    if res is not None: return res
            return None
            
        val = _find(data, key)
        if val is not None:
            return val
        raise KeyError(f"Key '{key}' not found in YAML (searched at root and nested levels)")
        
    current = data
    for key in keys_path:
        if key not in current:
            raise KeyError(f"Key '{key}' not found in path '{resolved_path}'")
        current = current[key]
        
    return current

def _get_all_perf_values(perf_dict: Dict) -> List[float]:
    """Extract all numeric values from nested perf dictionary."""
    vals = []
    for v in perf_dict.values():
        if isinstance(v, (int, float)):
            vals.append(float(v))
        elif isinstance(v, dict):
            vals.extend(_get_all_perf_values(v))
    return vals

def _filter(files: List[str], 
            inclusion: Optional[Tuple], 
            verbose: bool = False, 
            show_progress: bool = False) -> Tuple[List[int], List[Tuple[str, str, bool]]]:
    """Internal function to filter session files based on YAML metadata criteria."""
    if not inclusion:
        return list(range(len(files))), []
        
    valid_indices = []
    files_info = []
    
    shortcuts = {
        "sr": "acquisition.samplingrate", 
        "ref": "acquisition.reference", 
        "perfLHRH": "perf.left_hand-right_hand", 
        "perfRHF": "perf.right_hand-feet"
    }
    
    perf_filters_present = any(
        fp.startswith("perf") or shortcuts.get(fp, "").startswith("perf")
        for fp, _ in inclusion
    )
    
    if show_progress:
        print(f"\n{'─' * 65}\n🔍 Applying {len(inclusion)} filter(s) to {len(files)} session(s)...")
        
    for file_idx, file_path in enumerate(files):
        yml_path = os.path.splitext(file_path)[0] + ".yml"
        
        if not os.path.isfile(yml_path):
            files_info.append((file_path, "Missing YAML file", False))
            if show_progress: print(f"  ✗ {os.path.basename(file_path)}: Missing YAML file")
            continue
            
        with open(yml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
            
        if perf_filters_present:
            if "perf" not in yaml_data:
                files_info.append((file_path, "Auto-rejected: no 'perf' section in YAML", False))
                if show_progress: print(f"  ✗ {os.path.basename(file_path)}: Auto-rejected: no 'perf' section")
                continue
                
            perf_values = _get_all_perf_values(yaml_data["perf"])
            if perf_values and all(0 <= v <= 0.2 for v in perf_values):
                files_info.append((file_path, "Auto-rejected: all perf values ∈ [0, 0.2]", False))
                if show_progress: print(f"  ✗ {os.path.basename(file_path)}: Auto-rejected: all perf values ∈ [0, 0.2]")
                continue
                
        session_valid, status_msg = True, ""
        
        for filter_idx, (field_path, predicate) in enumerate(inclusion):
            try:
                value = _get_nested_value(yaml_data, field_path)
                if isinstance(value, list) and all(isinstance(x, str) for x in value):
                    value = CaseInsensitiveList(value)
                    
                if predicate(value):
                    if filter_idx == len(inclusion) - 1:
                        status_msg = f"Passed all {len(inclusion)} filter(s)"
                else:
                    session_valid = False
                    status_msg = f"Filter #{filter_idx + 1} failed: '{field_path}' = {value}"
                    break
            except Exception as e:
                session_valid = False
                status_msg = f"Error in filter #{filter_idx + 1} on '{field_path}': {str(e)}"
                break
                
        files_info.append((file_path, status_msg, session_valid))
        if show_progress:
            symbol = "✓" if session_valid else "✗"
            print(f"  {symbol} {os.path.basename(file_path)}: {status_msg}")
            
        if session_valid:
            valid_indices.append(file_idx)
            
    if show_progress:
        print(f"{'─' * 65}\n✓ Result: {len(valid_indices)}/{len(files)} session(s) passed all filters\n")
        
    return valid_indices, files_info



# ==============================================================================
# MAIN SELECTDB FUNCTION
# ==============================================================================
def selectDB(corpusDir: str,
             paradigm: str,
             classes: Optional[List[str]] = None,
             inclusion: Optional[Tuple] = None,
             summarize: bool = True,
             verbose: bool = False) -> List['InfoDB']:
    """
    Select BCI databases pertaining to the given BCI paradigm and all sessions
    meeting the provided inclusion criteria.
    
    Return the selected databases as a list of InfoDB structures, wherein the
    InfoDB.files field lists the included sessions only.
    
    Args:
        corpusDir: directory on local computer where to start the search
        paradigm: BCI paradigm to use ('P300', 'MI', or 'ERP')
        classes: labels of classes the databases must include
                 (default: ['target', 'nontarget'] for P300, None for MI/ERP)
        inclusion: tuple of custom filter conditions (field_path, lambda_function)
        summarize: if True, print a summary table of selected databases
        verbose: if True, print additional feedback
        
    Returns:
        List of InfoDB structures for selected databases
        
    Examples:
        >>> # Basic selection
        >>> DB_P300 = selectDB("/path/to/corpus", "P300")
        
        >>> # Selection with inclusion lambda filters (replaces minTrials)
        >>> inclusion_filters = (
        ...     ("sr", lambda x: x >= 256),
        ...     ("tpc", lambda x: min(x.values()) >= 50),  # Minimum trials per class
        ...     ("acquisition.sensors", lambda x: "Fz" in x)
        ... )
        >>> DB_MI = selectDB("/path/to/corpus", "MI", inclusion=inclusion_filters)
    """
    # Set default classes for P300
    if paradigm == "P300" and classes is None:
        classes = ["target", "nontarget"]
        
    # Auto-correct flat tuple format: ("field", lambda) → (("field", lambda),)
    if inclusion is not None and len(inclusion) > 0 and isinstance(inclusion[0], str):
        warnings.warn("Database.selectDB: `inclusion` was passed as a flat tuple — automatically wrapped. Add a trailing comma to avoid this: ((...),)")
        inclusion = (inclusion,)
    
    # Validate paradigm
    if paradigm not in ("MI", "P300", "ERP"):
        raise ValueError("Database.selectDB: Unsupported paradigm. Use 'MI', 'P300' or 'ERP'")
    
    # Check if there's a paradigm subfolder
    paradigmDir = os.path.join(corpusDir, paradigm)
    if os.path.isdir(paradigmDir):
        corpusDir = paradigmDir
    
    if not os.path.isdir(corpusDir):
        raise ValueError(f"Database.selectDB: invalid directory: {corpusDir}")
    
    dbDirs = getFoldersInDir(corpusDir)
    if not dbDirs:
        raise ValueError(f"Database.selectDB: No database found in: {corpusDir}")
    
    # Check paradigm and classes requirements
    if (paradigm in ("MI", "ERP")) and classes is None:
        print(f"Database.selectDB: No class filter specified for {paradigm} paradigm. "
              "All databases will be returned.")
        print("Info: If you plan to train ML models, specify 'classes' argument.")
    
    selectedDB = []  # List of InfoDB structures
    all_cLabels = set()  # All available classes
    db_filtering_info = []    # inclusion tracking: (database_name, files_info)
    n_class_match = 0
    
    # Normalize classes to lowercase
    norm_classes = None if classes is None else [c.lower() for c in classes]
    
    if verbose:
        print(f"Searching for {paradigm} databases" +
              (" (no class filter)" if classes is None else f" containing: {', '.join(classes)}"))
    
    for dbDir in dbDirs:
        info = infoDB(dbDir)
        
        # Skip if paradigm doesn't match
        if info.paradigm.upper() != paradigm:
            continue
        
        # Collect classes and check validity
        all_cLabels.update(info.cLabels)
        if classes is not None:
            if not all(req_class in [c.lower() for c in info.cLabels] for req_class in norm_classes):
                continue
                
        n_class_match += 1

        # Handle inclusion custom filters
        if inclusion is not None:
            # Remove paradigm and classes filters from inclusion if present
            forbidden = [f for (f, _) in inclusion if f in ("paradigm", "classes")]
            if forbidden:
                warnings.warn(f"Database.selectDB: Filters automatically removed (conflict with function arguments): {', '.join(forbidden)}")
                inclusion = tuple(filter(lambda x: x[0] not in ("paradigm", "classes"), inclusion))
                if not inclusion:
                    inclusion = None
                    
            if inclusion is not None:
                # Apply custom filters
                inc_valid_indices, files_info = _filter(info.files, inclusion, verbose=False, show_progress=False)
                
                if files_info:
                    db_filtering_info.append((info.dbName, files_info))
                    
                if not inc_valid_indices:
                    continue  # Skip database if no valid files
                    
                info.files = [info.files[i] for i in inc_valid_indices]
        
        selectedDB.append(info)
    
    if not selectedDB:
        if inclusion is not None and n_class_match > 0:
            raise ValueError(f"Database.selectDB: {n_class_match} {paradigm} database(s) matched paradigm/classes criteria but no session passed the `inclusion` filters.")
        else:
            avail_classes = ", ".join(sorted(all_cLabels)) if all_cLabels else "none"
            raise ValueError(f"Database.selectDB: No {paradigm} database contains all "
                             f"selected classes: {', '.join(classes) if classes else 'N/A'}\n"
                             f"Available classes: {avail_classes}")
                
    # Print excluded files info from inclusion filters
    if db_filtering_info:
        if verbose:
            print(f"\n{'═' * 65}")
            print("⚠️  FILTERING RESULTS BY DATABASE")
            print('═' * 65)
            for dbName, files_info in db_filtering_info:
                n_passed = sum(1 for _, _, passed in files_info if passed)
                n_total = len(files_info)
                print(f"\nDatabase: {dbName}")
                print('─' * 65)
                for file_path, status, passed in files_info:
                    symbol = "✓" if passed else "✗"
                    print(f"  {symbol} {os.path.basename(file_path)}: {status}")
                print('─' * 65)
                print(f"✓ Result: {n_passed}/{n_total} session(s) passed all filters")
        else:
            print(f"\n{'─' * 65}")
            print("⚠️  Files excluded by custom filters:")
            for dbName, files_info in db_filtering_info:
                excluded = [os.path.basename(f) for f, _, passed in files_info if not passed]
                if excluded:
                    print(f"  Database: {dbName}")
                    for f in excluded:
                        print(f"    • {f}")
            print('─' * 65, "\n")
    
    print()
    
    if verbose:
        print(f"\n{'═' * 50}")
        print(f"✓ {len(selectedDB)} database(s) selected (Database - Condition):")
        for db in selectedDB:
            print(f"  • {db.dbName} - {db.condition}")
        print('═' * 50)
    
# Create summary table
    if summarize:
        summary_data = []
        for db in selectedDB:
            min_sessions = min(db.nSessions)
            max_sessions = max(db.nSessions)
            nsessions_str = f"{min_sessions}" if min_sessions == max_sessions else f"({min_sessions},{max_sessions})"
            
            summary_data.append({
                'dbName': db.dbName,
                'condition': db.condition,
                'nSubjects': db.nSubjects,
                'nSessions': nsessions_str,
                'nSensors': db.nSensors,
                'sensorType': db.sensorType,
                'nClasses': db.nClasses,
                'sr': db.sr,
                'wl': db.wl,
                'os': db.offset
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        summary_df.index = summary_df.index
        
        print("SUMMARY TABLE OF SELECTED DATABASES")
        print('═' * 150)
        print(summary_df.to_string(index=True))  
        print('═' * 150)
        print("\n💡 For detailed trial counts per class, please inspect individual database structures")
    
    return selectedDB