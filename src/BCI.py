# Database module for EEG database management in NY format.
# v 0.1 Dec 2025
# Part of the Eegle package - Python version
# Copyright Fahim Doumi, CeSMA, Marco Congedo, CNRS, University Grenoble Alpes.
#
# ? ¤ CONTENT ¤ ? 
# 
# - CVres: dataclass holding information summarizing a cross-validation classification task 
# - encode: generate ovariance matrices based on EEG structure
# - crval: make a cross validation and return CVres structure

import numpy as np
import time
from dataclasses import dataclass
from pyriemann.estimation import Covariances, ERPCovariances
from sklearn.model_selection import StratifiedKFold
from typing import List, Union, Optional
from sklearn.metrics import balanced_accuracy_score
from InOut import EEG

@dataclass
class CVres:
    """
    Cross-validation results structure.
    
    Attributes:
        cvType: Type of cross-validation (e.g., "10-fold")
        scoring: Type of accuracy computed (e.g., "balanced_accuracy")
        modelType: Type of machine learning model used
        nTrials: Total number of trials
        accs: Accuracies for each fold
        avgAcc: Average accuracy across folds
        stdAcc: Standard deviation of accuracy across folds
        ms: Execution time in milliseconds
    """
    cvType: str
    scoring: Optional[str] = None
    modelType: Optional[str] = None
    nTrials: Optional[int] = None
    accs: Optional[np.ndarray] = None
    avgAcc: Optional[float] = None
    stdAcc: Optional[float] = None
    ms: Optional[int] = None
    
    def __repr__(self):
        """Pretty print for REPL output"""
        lines = [
            "\n◕ Cross-Validation Accuracy",
            "⭒  ⭒    ⭒       ⭒         ⭒",
            f".cvType   : {self.cvType}"
        ]
        
        if self.scoring is not None:
            lines.append(f".scoring  : {self.scoring}")
        if self.modelType is not None:
            lines.append(f".modelType: {self.modelType}")
        if self.nTrials is not None:
            lines.append(f".nTrials  : {self.nTrials}")
        if self.accs is not None:
            lines.append(f".accs     : acc per fold (access with .accs)")
        if self.avgAcc is not None:
            lines.append(f".avgAcc   : {self.avgAcc:.3f}")
        if self.stdAcc is not None:
            lines.append(f".stdAcc   : {self.stdAcc:.3f}")
        if self.ms is not None:
            lines.append(f".ms       : {self.ms}")
        
        return "\n".join(lines)

def encode(o: EEG, 
           paradigm: str = None,
           covtype: str = 'lwf',
           targetLabel: str = "target") -> np.ndarray:
    """
    Encode all trials in an EEG data structure as covariance matrices according to a given BCI paradigm.
    This is a simplified version using pyriemann for covariance estimation.
    
    Args:
        o: an instance of the EEG data structure containing trials and metadata
        paradigm: BCI paradigm, either 'ERP', 'P300', or 'MI'. 
                 By default uses the paradigm stored in o.paradigm
        covtype: covariance estimator for pyriemann (default: 'lwf')
                - 'scm': sample covariance matrix
                - 'lwf': Ledoit-Wolf shrinkage
                - 'oas': Oracle Approximating Shrinkage
                - 'mcd': Minimum Covariance Determinant
        targetLabel: label of the target class (for P300 paradigm only). 
                    Default is "target"
    
    Returns:
        ndarray: covariance matrices of shape (n_trials, n_channels, n_channels)
    
    Examples:
        # For P300 paradigm
        o = readNY("data.npz", bandPass=(1, 24))
        C = encode(o)
        
        # For MI paradigm  
        o = readNY("mi_data.npz", paradigm="MI", bandPass=(8, 32))
        C = encode(o, paradigm="MI")
    """
    
    # Validation
    if o.trials is None:
        raise ValueError("The EEG structure does not hold trials. "
                        "Make sure `classes` is not set to False when reading EEG data with readNY")
    
    # Use paradigm from EEG structure if not provided
    if paradigm is None:
        paradigm = o.paradigm
    
    paradigm = paradigm.upper()
    if paradigm not in ['ERP', 'P300', 'MI']:
        raise ValueError(f"Paradigm must be one of 'ERP', 'P300', 'MI'. Got: {paradigm}")
    
    # Convert trials list to numpy array format expected by pyriemann
    # pyriemann expects (n_trials, n_channels, n_samples)
    trials_array = np.array([trial.T for trial in o.trials])  # transpose each trial
    
    if paradigm == 'MI':
        # For Motor Imagery: direct covariance estimation without prototype
        cov_estimator = Covariances(estimator=covtype)
        covs = cov_estimator.transform(trials_array)
        
    elif paradigm == 'P300':
        # For P300: use target class as template for ERP covariances
        
        # Find target class index
        try:
            target_idx = [label.lower() for label in o.clabels].index(targetLabel.lower())
        except ValueError:
            raise ValueError(f"Target label '{targetLabel}' not found among class labels: {o.clabels}")
        
        # Create labels array for trials
        labels = np.array(o.y)
        
        # Use ERPCovariances with target class
        cov_estimator = ERPCovariances(classes=[target_idx + 1], estimator=covtype)
        covs = cov_estimator.fit_transform(trials_array, labels)
    return covs


def crval(clf,
          covs: Union[np.ndarray, List[np.ndarray]],
          labels: List[int],
          n_folds: int = 10,
          shuffle: bool = False,
          random_state: int = None,
          scoring: str = 'balanced_accuracy') -> CVres:
    """
    Cross-validation for covariance matrices using pyRiemann classifiers.
   
    Args:
        clf: pyRiemann classifier instance (MDM, FgMDM, TSclassifier, SVC, etc.)
        covs: covariance matrices (n_trials, n_channels, n_channels) or list of matrices
        labels: class labels for each trial
        n_folds: number of folds for cross-validation (default: 10)
        shuffle: whether to shuffle data before splitting (default: False)
        random_state: random state for reproducible shuffling (default: None)
        scoring: type of accuracy to compute (default: 'balanced_accuracy')
        verbose: print debug information (default: False)
   
    Returns:
        CVres: Cross-validation results structure
   
    Examples:
        # MDM classifier
        from pyriemann.classification import MDM
        clf = MDM(metric='riemann', n_jobs=4)
        results = crval(clf, covs, labels, n_folds=10)
        print(results)
        print(f"Mean accuracy: {results.avgAcc:.3f} ± {results.stdAcc:.3f}")
       
        # SVM with linear kernel
        from pyriemann.classification import SVC
        clf = SVC(kernel='linear')
        results = crval(clf, covs, labels)
        
        # Tangent Space + Logistic Regression (Lasso)
        from pyriemann.classification import TSclassifier
        from sklearn.linear_model import LogisticRegression
        clf = TSclassifier(clf=LogisticRegression(penalty='l1', solver='saga'))
        results = crval(clf, covs, labels, n_folds=5, shuffle=True, random_state=42)
    """
    
    # Get model type name
    model_type = clf.__class__.__name__
    
    # Initialize arrays
    acc = np.zeros(n_folds)
   
    # Setup cross-validation
    cv_params = {'n_splits': n_folds, 'shuffle': shuffle}
    if shuffle and random_state is not None:
        cv_params['random_state'] = random_state

    cv = StratifiedKFold(**cv_params)

    # Timer start
    start = time.perf_counter()
    
    # Cross-validation loop
    for j, (train_index, test_index) in enumerate(cv.split(covs, labels)):
        # Clone classifier for this fold
        
        # Splitting
        C_train = [covs[i] for i in train_index]
        C_test = [covs[i] for i in test_index] 
        y_train = [labels[i] for i in train_index]
        y_true = [labels[i] for i in test_index]

        # Fit classifier
        clf.fit(np.array(C_train), y_train)

        # Predict labels
        y_pred = clf.predict(np.array(C_test))

        # Calculate balanced accuracy
        acc[j] = balanced_accuracy_score(y_true, y_pred)

    exec_time_ms = int((time.perf_counter() - start) * 1000)
   
    # Create result structure
    cv_type = f"{n_folds}-fold"
    if shuffle:
        cv_type += " (shuffled)"
    
    results = CVres(
        cvType=cv_type,
        scoring=scoring,
        modelType=model_type,
        nTrials=len(labels),
        accs=acc,
        avgAcc=float(np.mean(acc)),
        stdAcc=float(np.std(acc)),
        ms=exec_time_ms
    )
    
    print(f"\nDone in {exec_time_ms} ms")
   
    return results