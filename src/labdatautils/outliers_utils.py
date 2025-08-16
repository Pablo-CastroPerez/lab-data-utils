import numpy as np
from scipy.stats import t


#-----------------------------------------Helper function------------------------------------------------

# Remove invalid data, (NaN/Inf)
def filter_finite(x, verbose=True):
    arr = np.asarray(x, dtype=float).ravel()
    finite_mask = np.isfinite(arr)
    if verbose:
        bad = np.flatnonzero(~finite_mask)
        if bad.size:
            print(f"Excluded non-finite positions: {', '.join(map(str, bad))} (NaN/Inf)")
    return arr[finite_mask]         

#----------------------------------------------z-score----------------------------------------------------

def compute_zscores(x, *, ddof: int = 0, verbose: bool = True):
   
    xf = filter_finite(x, verbose=verbose) # Clean data
    if xf.size == 0:
        return np.array([])

    mu = xf.mean()
    sd = xf.std(ddof=ddof)
    z = np.zeros_like(xf) if sd == 0 else (xf - mu) / sd

    return z


def detect_outliers_zscore(x, threshold: float = 3.0, ddof: int = 0, verbose: bool = True):
 
    xf = filter_finite(x, verbose = verbose)
    z = compute_zscores(x, ddof=ddof, verbose= False)

    if z.size == 0:
        if verbose:
            print("No finite data points available; nothing to detect.")
        return np.array([], dtype=bool)  

    outlier_mask = np.abs(z) > threshold # Criteria for an outlier, defaulted to threshold = 3 if not specified 

    # Optionally report detected outliers 
    if verbose and np.any(outlier_mask):
        print("Outliers detected:")
        for val, z_val in zip(xf[outlier_mask], z[outlier_mask]):
            print(f"  value={val:g}, z={z_val:.2f}")

    return outlier_mask


def remove_outliers_zscore(x, threshold: float = 3.0, ddof: int = 0, verbose: bool = True):
    
    xf = filter_finite(x,verbose=verbose)
    outlier_mask = detect_outliers_zscore(x, threshold = threshold, ddof =  ddof, verbose = False) 
    z_scores = compute_zscores(x, ddof=ddof, verbose = False)

    # Optionally report exclused values (indeces refered to the finite valued data, i.e. after filter_finite)
    if verbose and np.any(outlier_mask):
        print("Outliers removed (z-score):")
        outlier_indices = np.where(outlier_mask)[0]  # Indices refer to the finite valued data, i.e. after filter_finite
        for idx in outlier_indices:
            print(f"  clean_data[{idx}] = {xf[idx]:g}, z[{idx}] = {z_scores[idx]:.2f}")

    return xf[~outlier_mask]

#------------------------------------------------MAD------------------------------------------------------

def compute_mad_scores(x, verbose: bool = True):

    xf = filter_finite(x, verbose=verbose) # Clean data
    if xf.size == 0:
        return np.array([])
    median_val = np.median(xf)
    mad = np.median(np.abs(xf - median_val))
    if mad == 0:
        return np.zeros_like(xf)
    return 0.67448975 * (xf - median_val) / mad


def detect_outliers_mad(x, threshold: float = 3.5, verbose: bool = True):

    xf = filter_finite(x, verbose = verbose)
    modified_z = compute_mad_scores(xf, verbose=False)  

    if modified_z.size == 0:
        if verbose:
            print("No finite data points available; nothing to detect.")
        return np.array([], dtype=bool) 

    outlier_mask = np.abs(modified_z) > threshold # Criteria for an outlier, defaulted to threshold = 3.5 if not specified 

    # Optionally report detected outliers 
    if verbose and np.any(outlier_mask):
        print("Outliers detected (MAD):")
        for val, m_val in zip(xf[outlier_mask], modified_z[outlier_mask]):
            print(f"  value={val:g}, z={m_val:.2f}")

    return  outlier_mask

def remove_outliers_mad(x, threshold: float = 3.5, verbose: bool = True):
    
    xf = filter_finite(x,verbose=verbose)
    outlier_mask = detect_outliers_mad(xf, threshold = threshold, verbose = False)
    m_value = compute_mad_scores(xf, verbose = False)

    # Optionally report excluded outliers 
    if verbose and np.any(outlier_mask):
        print("Outliers removed (MAD):")
        outlier_indices = np.where(outlier_mask)[0]  # Indices refer to the finite valued data, i.e. after filter_finite
        for idx in outlier_indices:
            print(f"  clean_data[{idx}] = {xf[idx]:g}, z[{idx}] = {m_value[idx]:.2f}")

    return xf[~outlier_mask]


#------------------------------------------------IQR------------------------------------------------------

def compute_iqr_scores(x, *, threshold: float = 1.5, verbose: bool = True):

    xf = filter_finite(x, verbose=verbose)
    if xf.size == 0:
        return np.array([])

    q1 = np.percentile(xf, 25)
    q3 = np.percentile(xf, 75)
    iqr = q3 - q1

    
    if iqr == 0:
        return np.zeros_like(xf)

    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr

    scores = np.zeros_like(xf)
    below = xf < lower
    above = xf > upper
    scores[below] = (lower - xf[below]) / iqr
    scores[above] = (xf[above] - upper) / iqr

    return scores


def detect_outliers_iqr(x, threshold: float = 1.5, verbose: bool = True):
    
    xf = filter_finite(x, verbose = verbose)
    iqr_scores = compute_iqr_scores(xf, threshold=threshold, verbose=False)
    if iqr_scores.size == 0:
        if verbose:
            print("No finite data points available; nothing to detect.")
        return np.array([], dtype=bool) 

    outlier_mask = iqr_scores > 0. # Every value outside the range [q1 - threshold * iqr ,  q3 + threshold * iqr] is an outlier

    # Optionally report detected outliers 
    if verbose and np.any(outlier_mask):
        print("Outliers detected (IQR):")
        for  val, sc in zip(xf[outlier_mask], iqr_scores[outlier_mask]):
            print(f"  value={val:g}, z={sc:.2f}")

    return outlier_mask


def remove_outliers_iqr(x, threshold: float = 1.5, verbose: bool = True):
  
  xf = filter_finite(x)
  outlier_mask = detect_outliers_iqr(xf, threshold= threshold, verbose=False)
  iqr_scores = compute_iqr_scores(xf, threshold=threshold, verbose=False)

  # Optionally report detected outliers   
  if verbose and np.any(outlier_mask):
        print("Outliers removed (IQR):")
        outlier_indices = np.where(outlier_mask)[0]  # Indices refer to the finite valued data, i.e. after filter_finite
        for idx in outlier_indices:
            print(f"  clean_data[{idx}] = {xf[idx]:g}, z[{idx}] = {iqr_scores[idx]:.2f}")
  return xf[~outlier_mask]

#----------------------------------------------Grubb's----------------------------------------------------


def grubbs_crit(n: int, alpha: float = 0.05) -> float:
   
   # Calculate t-value using the percent point function from scipy 
    df = n - 2
    tcrit = t.ppf(1 - alpha/(2*n), df)
    return (n - 1) / np.sqrt(n) * np.sqrt(tcrit**2 / (df + tcrit**2))


def compute_grubbs_scores(x, *, ddof: int = 1, verbose: bool = True):

    xf = filter_finite(x, verbose=verbose)
    if xf.size == 0:
        return np.array([])
    s = xf.std(ddof=ddof)
    return np.zeros_like(xf) if s == 0 else np.abs((xf - xf.mean()) / s)

def detect_outliers_grubbs(x, alpha: float = 0.05, ddof: int = 1, verbose: bool = True):

    xf = filter_finite(x, verbose=verbose)
    G = compute_grubbs_scores(xf, ddof=ddof, verbose=False)  
    if G.size == 0:
        if verbose:
            print("No finite data points available; nothing to detect.")
        return np.array([], dtype=bool)
    
    Gcrit = grubbs_crit(xf.size, alpha)
    outlier_mask = G > Gcrit
    if verbose and np.any(outlier_mask):
        print("Outliers detected (Grubbs):")
        for i in np.where(outlier_mask)[0]:
            print(f"  value={xf[i]:g}, G={G[i]:.3f} (Gcrit={Gcrit:.3f})")
    return outlier_mask

def remove_outliers_grubbs(x, alpha: float = 0.05, ddof: int = 1, verbose: bool = True):

    xf = filter_finite(x, verbose=verbose)
    outlier_mask = detect_outliers_grubbs(xf, alpha=alpha, ddof=ddof, verbose=False)
    G = compute_grubbs_scores(xf, ddof=ddof, verbose=False)

    if verbose and np.any(outlier_mask):
        print("Outliers removed (Grubbs):")
        for idx in np.where(outlier_mask)[0]:
            print(f"  clean_data[{idx}] = {xf[idx]:g}, G[{idx}] = {G[idx]:.3f}")

    return xf[~outlier_mask]


#---------------------------------------------General---------------------------------------------------

def detect_outliers(x, method: str, **kwargs):

    method = method.lower()
    if method in ('z', 'zscore', 'z-score'):
        mask = detect_outliers_zscore(x, **kwargs)
      
    elif method == 'mad':
        mask = detect_outliers_mad(x, **kwargs)
       
    elif method == 'iqr':
        mask = detect_outliers_iqr(x, **kwargs)
        
    elif method == 'grubbs':
        mask = detect_outliers_grubbs(x, **kwargs)
        
    else:
        raise ValueError("method must be one of: 'z','mad','iqr','grubbs'")
    
    return mask 

def remove_outliers(x, method: str, **kwargs):

    method = method.lower()
    if method in ('z', 'zscore', 'z-score'):
        clean = remove_outliers_zscore(x, **kwargs)

    elif method == 'mad':
        clean = remove_outliers_mad(x, **kwargs)

    elif method == 'iqr':
        clean = remove_outliers_iqr(x, **kwargs)

    elif method == 'grubbs':
        clean = remove_outliers_grubbs(x, **kwargs)

    else:
        raise ValueError("method must be one of: 'z','mad','iqr','grubbs'")

    return clean
