# pyLittleEegle: Python package for FII BCI Corpus

**Version:** 0.1 (Dec 2025)  
**Authors:** Fahim Doumi (CeSMA, University Federico II, Naples),
Marco Congedo (CNRS, University Grenoble Alpes, Grenoble)

This repository contains a suite of Python tools designed to manage, process, and analyze EEG data from the [**FII BCI Corpus**](https://marco-congedo.github.io/Eegle.jl/dev/documents/FII%20BCI%20Corpus%20Overview/), specifically formatted in the [**NY format**](https://marco-congedo.github.io/Eegle.jl/dev/documents/NY%20format/).

**pyLittleEegle** is a pure-Python, **PyRiemann-friendly** port of the core BCI functionalities from the Julia [Eegle](https://github.com/Marco-Congedo/Eegle.jl) package.
It leverages the Python scientific ecosystem (`numpy`, `pandas`, `scikit-learn`, `pyriemann`, `scipy`, etc.) to streamline the management of **NY-format** databases and facilitate BCI classification pipelines.

## Core Dependencies

To use these tools, you will need the following libraries:

* `numpy`
* `pandas`
* `scikit-learn`
* `pyriemann`
* `scipy`
* `pyyaml`

## Modules Overview

The toolkit is divided into three main modules, designed to be used sequentially:

### 1. Database Management (`Database.py`)

*The Librarian.*

This module handles the exploration and selection of datasets without loading all data into memory. It allows you to filter the massive FII BCI Corpus based on specific criteria.

**Key Features:**

* **`infoDB` Structure:** An immutable dataclass that summarizes all metadata of a database (Subject count, sampling rate, paradigms, hardware, etc.).
* **`selectDB`:** The main entry point. It scans directories to find databases matching your requirements.
    * *Filter by Paradigm:* Select only 'MI' (Motor Imagery), 'P300', or 'ERP'.
    * *Filter by Class:* Keep only databases containing specific classes (e.g., `["right_hand", "feet"]`).
    * *Filter by Minimum number of trial per class:* Exclude sessions that do not have enough trials via the `minTrials` argument.

**Example:**
```python
from Database import selectDB

# Find all Motor Imagery databases containing "right_hand" and "feet"
# with at least 20 trials per class.
DBs = selectDB(
    corpusDir="./Data", 
    paradigm="MI", 
    classes=["right_hand", "feet"], 
    minTrials=20
)
```

### 2\. Input/Output & Preprocessing (`InOut.py`)

*The Loader.*

Once a database is selected, this module reads the actual recordings (`.npz` and `.yml` files) and structures them for analysis.

**Key Features:**

  * **`readNY`:** The core function to load an EEG recording.
      * **Filtering:** Can apply BandPass or BandStop filters on the fly.
      * **Class Selection:** Can load specific classes (e.g., ignore "rest" and keep only active tasks).
      * **Standardization (`stdClass`):** Automatically maps class labels to a standard numerical convention (e.g., `right_hand` -\> 2), facilitating transfer learning across different datasets.
  * **`EEG` Structure:** A comprehensive dataclass containing the signal matrix (`.X`), the stimulation vector (`.stim`), trial markers, and all acquisition metadata.

**Example:**

```python
from InOut import readNY

# Load a specific session, apply a 8-30Hz bandpass filter, 
# and keep only right_hand and feet classes.
o = readNY(
    "path/to/subject01_session01.npz",
    bandPass=(8, 32),
    classes=["right_hand", "feet"]
)
```

### 3\. BCI Classification (`BCI.py`)

*The Analyst.*

This module provides tools to encode EEG data into geometric features (Covariance Matrices) and perform classification using Riemannian Geometry.

**Key Features:**

  * **`encode`:** Converts the raw EEG trials from the `EEG` structure into Covariance Matrices.
      * **Seamless Integration:** It explicitly handles the data formatting logic to ensure compatibility with the Python ecosystem. It automatically reshapes and transposes the internal data into the standard **`(n_trials, n_channels, n_samples)`** `numpy` array format strictly required by `pyriemann` and `scikit-learn`, abstracting away the manual formatting usually needed.
      * **Versatility:** Supports different estimators (SCM, LWF, OAS) and paradigms (ERPCovariances for P300, direct Covariances for MI).
  * **`crval`:** A streamlined Cross-Validation wrapper. It takes a classifier (e.g., from `pyriemann` or `sklearn`), the covariance matrices, and labels, then returns a `CVres` summary object containing accuracy metrics (balanced accuracy, mean, std) and execution time.

**Example:**

```python
from BCI import encode, crval
from pyriemann.classification import MDM

# 1. Encode data to Covariance Matrices
# Automatically handles the reshape to (n_trials, n_channels, n_samples)
covs = encode(o, paradigm="MI")

# 2. Define a Classifier (Minimum Distance to Mean)
clf = MDM(metric='riemann')

# 3. Run Cross-Validation
results = crval(clf, covs, o.y, n_folds=10)
display(results)
```

## 📚 Tutorial

A complete step-by-step workflow is available in **`FullTutorial.ipynb`**.

This notebook demonstrates the entire pipeline:

1.  Selecting databases using `Database.py`.
2.  Loading and preprocessing data with `InOut.py`.
3.  Encoding and classifying signals using `BCI.py`.

-----

*Copyright © 2025 Fahim Doumi, CeSMA, University Federico II, Marco Congedo, CNRS, University Grenoble Alpes.*
