# Database module for EEG database management in NY format.
# v 0.1 Dec 2025
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
from typing import List, Dict, Optional
import yaml
import pandas as pd
import numpy as np

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
        
        # Format ntrials_per_class - show mean ± std + min,max
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
        nTrials.append(stim["trials_per_class"])
        
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


def selectDB(corpusDir: str,
             paradigm: str,
             classes: Optional[List[str]] = None,
             minTrials: Optional[int] = None,
             summarize: bool = True,
             verbose: bool = False) -> List[InfoDB]:
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
        minTrials: minimum number of trials for all classes in sessions
        summarize: if True, print a summary table of selected databases
        verbose: if True, print additional feedback
        
    Returns:
        List of InfoDB structures for selected databases
        
    Examples:
        >>> DB_P300 = selectDB("/path/to/corpus", "P300")
        >>> DB_MI = selectDB("/path/to/corpus", "MI", classes=["left_hand", "right_hand"])
        >>> DB = selectDB("/path/to/corpus", "MI", classes=["rest", "feet"], minTrials=50)
    """
    # Set default classes for P300
    if paradigm == "P300" and classes is None:
        classes = ["target", "nontarget"]
    
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
    excluded_files_info = []  # (database_name, excluded_files)
    
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
        
        # Handle minTrials filtering
        if minTrials is not None:
            excluded_files = []
            valid_indices = []
            classes_to_check = info.cLabels if classes is None else classes
            
            for file_idx, file_path in enumerate(info.files):
                session_valid = True
                for class_name in classes_to_check:
                    # Find actual class name (case-insensitive)
                    if classes is None:
                        actual_class = class_name
                    else:
                        actual_class_list = [c for c in info.cLabels 
                                           if c.lower() == class_name.lower()]
                        actual_class = actual_class_list[0] if actual_class_list else None
                    
                    if actual_class and actual_class in info.nTrials:
                        if info.nTrials[actual_class][file_idx] < minTrials:
                            session_valid = False
                            break
                
                if session_valid:
                    valid_indices.append(file_idx)
                else:
                    excluded_files.append(file_path)
            
            # Skip database if no valid files
            if not valid_indices:
                if excluded_files:
                    excluded_files_info.append((info.dbName, excluded_files))
                continue
            
            # Filter files if some were excluded
            if excluded_files:
                excluded_files_info.append((info.dbName, excluded_files))
                info.files = [info.files[i] for i in valid_indices]
        
        selectedDB.append(info)
    
    if not selectedDB:
        avail_classes = ", ".join(sorted(all_cLabels)) if all_cLabels else "none"
        raise ValueError(f"Database.selectDB: No {paradigm} database contains all "
                        f"selected classes: {', '.join(classes) if classes else 'N/A'}\n"
                        f"Available classes: {avail_classes}")
    
    # Print excluded files info
    if excluded_files_info:
        print(f"\n{'─' * 65}")
        print(f"⚠️  Files excluded due to insufficient trials per class (< {minTrials}):")
        for dbName, files in excluded_files_info:
            print(f"\n  Database: {dbName}")
            for file in files:
                print(f"    • {os.path.basename(file)}")
    
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
        print("SUMMARY TABLE OF SELECTED DATABASES")
        print('═' * 150)
        print(summary_df.to_string(index=True))
        print('═' * 150)
        print("\n💡 For detailed trial counts per class, please inspect individual database structures")
    
    return selectedDB