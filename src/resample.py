from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler

def re_sample(X, y, func='random_under_sample', random_state=None):
    """
    Apply resampling techniques to the dataset to address class imbalance.

    This function takes a dataset and applies one of several resampling techniques to it, 
    based on the 'func' argument provided. Resampling techniques include both oversampling 
    and undersampling methods. The function supports random oversampling, Synthetic Minority 
    Over-sampling Technique (SMOTE), Adaptive Synthetic (ADASYN), BorderlineSMOTE, 
    KMeansSMOTE, ClusterCentroids, and random undersampling.

    Parameters:
    - X: Feature dataset (usually a DataFrame or a 2D array).
    - y: Target values associated with X.
    - func (str, optional): The resampling technique to apply. Supported values are 
      'random_over_sample', 'SMOTE', 'ADASYN', 'BorderlineSMOTE', 'KMeansSMOTE', 
      'ClusterCentroids', and 'random_under_sample'. If None, no resampling is applied.
    - random_state (int, optional): The random state for reproducibility.

    Returns:
    - X_resampled, y_resampled: The resampled feature set and target values. If 'func' is None,
      the function returns None.

    If an unsupported 'func' value is provided, the function returns None.
    """
    
    if func == None:
        return
    
    elif func == 'random_over_sample':
        ros = RandomOverSampler(random_state=random_state)
        X_resampled, y_resampled = ros.fit_resample(X, y)
    
    elif func == 'SMOTE':
        X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    
    elif func == 'ADASYN':
        X_resampled, y_resampled = ADASYN().fit_resample(X, y)
        
    elif func == 'BorderlineSMOTE':
        X_resampled, y_resampled = BorderlineSMOTE().fit_resample(X, y)
        
    elif func == 'KMeansSMOTE':
        X_resampled, y_resampled = KMeansSMOTE(cluster_balance_threshold=0.005).fit_resample(X, y) # https://arxiv.org/pdf/1711.00837.pdf
    
    elif func == 'ClusterCentroids':
        X_resampled, y_resampled = ClusterCentroids().fit_resample(X, y)
        
    elif func == 'random_under_sample':
        X_resampled, y_resampled = RandomUnderSampler(random_state=random_state).fit_resample(X, y) 
    
    else:
        return

    return X_resampled, y_resampled