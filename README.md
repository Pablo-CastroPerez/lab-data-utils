# lab-data-utils  
**Laboratory Data Utilities**

A Python library for **experimental data analysis and reporting**.  
Includes tools for **outlier detection**,  **error propagation** and **LaTeX-ready table formatting**.

## Features

- **Error propagation** via finite difference derivatives, with correct significant figures.
- **LaTeX table generation** with support for merged value ± error columns.
- Designed for **clarity** and **reproducibility** in lab reports.

## Installation

```bash
pip install labdatautils
```
---

## Function Reference

### 1. `propagate(func, *args)`

**Arguments:**
- `func` *(callable)* — Function whose result and uncertainty are to be calculated.
- `*args` — Values and uncertainties, alternating: `val1, err1, val2, err2, ...`  
  Each can be a float, list, or NumPy array (all arrays must have the same length).

**Returns:**
- `values` *(ndarray)* — Computed function results.
- `errors` *(ndarray)* — Propagated uncertainties.

---

### 2. `generate_latex_table(column_headers, data_columns, **kwargs)`

**Arguments:**
- `column_headers` *(list of str)* — Ordered list of column names to display.
- `data_columns` *(sequence of arrays)* — One array per column, in the same order as `column_headers`.  
  Each array should contain the values for that column.
- `caption` *(str, optional)* — Table caption. Default: `"Data Table"`.
- `label` *(str, optional)* — LaTeX label for referencing. Default: `"tab:data"`.
- `error_map` *(dict or list of str, optional)* — Maps error columns to their associated value column.  
  Example: `{"dx": "x", "dy": "y"}` or `["dx:x", "dy:y"]`.
- `merge_error_style` *(str, optional)* — How to display value and error pairs:  
  `"separate"` (two columns), `"pm"` (e.g. `1.23 ± 0.04`), `"paren"` (e.g. `1.23(4)`). Default: `"separate"



### 3. Outlier Detection

The library provides three layers of functionality for multiple methods, z-score, modified (MAD) z-score, IQR and Grubb's test:

---

#### a) `compute_*` (e.g. `compute_zscores(x, **kwargs)`)

**Arguments:**
- `x` *(array-like)* — Input data.  
- `**kwargs` — Method-specific options (e.g. `ddof` for z-scores, `threshold` for IQR).  

**Returns:**
- `scores` *(ndarray)* — Raw scores for each point (z-scores, MAD-scores, IQR-scores, or Grubbs’ G values).  
  
---

#### b) `detect_outliers_*` (e.g. `detect_outliers_zscore(x, **kwargs)`)

**Arguments:**
- `x` *(array-like)* — Input data.  
- `threshold` *(float, optional)* — Criterion for outliers. Default depends on method  
  (`3.0` for z, `3.5` for MAD, `1.5` for IQR, significance level `α=0.05` for Grubbs).  
- `verbose` *(bool, optional)* — If `True`, prints detected outliers. Default: `True`.  

**Returns:**
- `mask` *(ndarray of bool)* — Boolean array, `True` where the input is an outlier.  

---

#### c) `remove_outliers_*` (e.g. `remove_outliers_iqr(x, **kwargs)`)

**Arguments:**
- `x` *(array-like)* — Input data.  
- `threshold` *(float, optional)* — Same meaning as in `detect_outliers_*`.  
- `verbose` *(bool, optional)* — If `True`, prints which points were removed. Default: `True`.  

**Returns:**
- `cleaned` *(ndarray)* — Input data with outliers removed.  

---

#### General entry points

Instead of calling method-specific functions, you can use the general function which have the different methods as arguments, for example:

```python
from lab_data_utils import detect_outliers, remove_outliers

mask = detect_outliers(data, method="iqr", threshold=1.5)
cleaned = remove_outliers(data, method="iqr", threshold=1.5)
