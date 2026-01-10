<p align="center">
  <img src="docs/src/assets/full logo.png" width="400" title="pyLittleEegle Logo">
</p>

# pyLittleEegle: Python package for FII BCI Corpus

**Version:** 0.1.1 (Jan 2026)  
**Authors:** Fahim Doumi (CeSMA, University Federico II, Naples),
Marco Congedo (CNRS, University Grenoble Alpes, Grenoble)

This repository contains a suite of Python tools designed to manage, process, and analyze EEG data from the [**FII BCI Corpus**](https://marco-congedo.github.io/Eegle.jl/dev/documents/FII%20BCI%20Corpus%20Overview/), specifically formatted in the [**NY format**](https://marco-congedo.github.io/Eegle.jl/dev/documents/NY%20format/).

**pyLittleEegle** is a pure-Python, **pyRiemann-friendly** and **MNE-free** port of the core BCI functionalities from the Julia [Eegle](https://github.com/Marco-Congedo/Eegle.jl) package.
It leverages the Python scientific ecosystem to streamline the management of **NY-format** databases and facilitate BCI classification pipelines.

## Installation

You can install `pyLittleEegle` directly from using pip:

```bash
pip install pylittleeegle

```

## Core Dependencies

To use these tools, you will need the following libraries (automatically installed when using pip):

* `numpy`
* `pandas`
* `scikit-learn`
* `pyriemann`
* `scipy`
* `pyyaml`

## Modules Overview

The toolkit is divided into three main modules, designed to be used sequentially:

### 1. Database Management (`ple.Database`)

*The Librarian.*

This module handles the exploration and selection of datasets without loading all data into memory. It allows you to filter the massive FII BCI Corpus based on specific criteria.

**Key Features:**

* **`InfoDB` Structure:** An immutable dataclass that summarizes all metadata of a database (Subject count, sampling rate, paradigms, hardware, etc.).
* **`selectDB`:** The main entry point. It scans directories to find databases matching your requirements.
* *Filter by Paradigm:* Select only 'MI' (Motor Imagery), 'P300', or 'ERP'.
* *Filter by Class:* Keep only databases containing specific classes (e.g., `["right_hand", "feet"]`).
* *Filter by Minimum number of trial per class:* Exclude sessions that do not have enough trials via the `minTrials` argument.

**Example:**

```python
import pylittleeegle as ple

# Find all Motor Imagery databases containing "right_hand" and "feet"
# with at least 20 trials per class.
DBs = ple.selectDB(
    corpusDir="./Data", 
    paradigm="MI", 
    classes=["right_hand", "feet"], 
    minTrials=20
)

```

### 2. Input/Output & Preprocessing (`ple.InOut`)

*The Loader.*

Once a database is selected, this module reads the actual recordings (`.npz` and `.yml` files) and structures them for analysis.

**Key Features:**

* **`readNY`:** The core function to load an EEG recording.
* **Filtering:** Can apply BandPass or BandStop filters on the fly.
* **Class Selection:** Can load specific classes (e.g., ignore "rest" and keep only active tasks).
* **Standardization (`stdClass`):** Automatically maps class labels to a standard numerical convention (e.g., `right_hand` -> 2), facilitating transfer learning across different datasets.

* **`EEG` Structure:** A comprehensive dataclass containing the signal matrix (`.X`), the stimulation vector (`.stim`), trial markers, and all acquisition metadata.

**Example:**

```python
import pylittleeegle as ple

# Load a specific session, apply a 8-30Hz bandpass filter, 
# and keep only right_hand and feet classes.
o = ple.readNY(
    "path/to/subject_01_session_01.npz",
    bandPass=(8, 32),
    classes=["right_hand", "feet"]
)

```

### 3. BCI Classification (`ple.BCI`)

*The Analyst.*

This module provides tools to encode EEG data into geometric features (Covariance Matrices) and perform classification using Riemannian Geometry.

**Key Features:**

* **`encode`:** Converts the raw EEG trials from the `EEG` structure into Covariance Matrices.
* **Seamless Integration:** It explicitly handles the data formatting logic to ensure compatibility with the Python ecosystem. It automatically reshapes and transposes the internal data into the standard **`(n_trials, n_channels, n_samples)`** `numpy` array format strictly required by `pyriemann` and `scikit-learn`, abstracting away the manual formatting usually needed.
* **Versatility:** Supports different estimators (SCM, LWF, OAS) and paradigms (ERPCovariances for P300, direct Covariances for MI).

* **`crval`:** A streamlined Cross-Validation wrapper. It takes a classifier (e.g., from `pyriemann` or `sklearn`), the covariance matrices, and labels, then returns a `CVres` summary object containing accuracy metrics (balanced accuracy, mean, std) and execution time.

**Example:**

```python
import pylittleeegle as ple
from pyriemann.classification import MDM

# 1. Encode data to Covariance Matrices
# Automatically handles the reshape to (n_trials, n_channels, n_samples)
covs = ple.encode(o, paradigm="MI")

# 2. Define a Classifier (Minimum Distance to Mean)
clf = MDM(metric='riemann')

# 3. Run Cross-Validation
results = ple.crval(clf, covs, o.y, n_folds=10)
display(results)

```

## Documentation & Eegle.jl Heritage

Most functions in this package are exact or lightweight ports of functionalities from the original Julia **[Eegle.jl package](https://marco-congedo.github.io/Eegle.jl/dev/)**.

* **Theoretical logic:** Since the logic and parameters remain consistent between the two versions, you can refer to the **Eegle.jl documentation** for in-depth theoretical explanations and methodological details.
* **Implementation details:** For Python-specific implementation details (arguments, return types, and syntax), please refer directly to the **source code files**. All functions are fully documented with extensive comments and docstrings.

## 📚 Tutorial

A complete step-by-step workflow is available in **`FullTutorial.ipynb`**.

This notebook demonstrates the entire pipeline:

1. Selecting databases using `ple.Database`.
2. Loading and preprocessing data with `ple.InOut`.
3. Encoding and classifying signals using `ple.BCI`.

---

*Copyright © 2025 Fahim Doumi, CeSMA, University Federico II, Marco Congedo, CNRS, University Grenoble Alpes.*
