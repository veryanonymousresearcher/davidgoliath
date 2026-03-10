import pandas as pd
import numpy as np


def max_case_duration_days(df, time_col, case_col, mask=None):
    """
    Compute the maximum (max time - min time) per case, in days.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    time_col : str
        Name of the time column
    case_col : str
        Name of the case id column
    mask : array-like of bool, optional
        Boolean mask with same length as df. Rows where mask is True
        are ignored in the calculation.

    Returns
    -------
    int
        Maximum duration in days across all cases
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    if mask is not None:
        assert len(mask) == len(df), "mask must have the same length as df"
        mask = np.asarray(mask, dtype=bool)
        df = df.loc[~mask]

    if df.empty:
        return 0

    durations = (
        df.groupby(case_col)[time_col]
          .agg(lambda x: (x.max() - x.min()).days)
    )

    return int(durations.max())


def dataset_statistics(
    df_orig: pd.DataFrame,
    df_filtered: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    val: pd.DataFrame | None,
    tt_st: str,
    tv_st: str,
    case_col: str,
    time_col: str,
    #test_split: str,
    val_split: str,
    tt_trunc: pd.DataFrame | None = None,
    tv_trunc: pd.DataFrame | None = None,
    train_target_mask: np.ndarray | None = None,
    test_target_mask: np.ndarray | None = None,
    val_target_mask: np.ndarray | None = None,  
    ):
    
    if train_target_mask is not None:
        train_info = train[~train_target_mask]
    else:
        train_info = train
    if test_target_mask is not None:
        test_info = test[~test_target_mask]
    else:
        test_info = test
    if val_target_mask is not None:
        val_info = val[~val_target_mask]
    else:
        val_info = val  
        
        
    dataset_info = {}
    dataset_info["orig_dataset_cases"] = df_orig[case_col].nunique()
    dataset_info["orig_dataset_events"] = len(df_orig)
    dataset_info["orig_dataset_start"] = df_orig[time_col].min().strftime("%Y-%m-%d")
    dataset_info["orig_dataset_end"] = df_orig[time_col].max().strftime("%Y-%m-%d")
    dataset_info["orig_dataset_max_duration_days"] = max_case_duration_days(df_orig, time_col, case_col)
    #if test_split == "prefix":
    dataset_info["filtered_dataset_cases"] = df_filtered[case_col].nunique()
    dataset_info["filtered_dataset_events"] = len(df_filtered)
    dataset_info["filtered_dataset_start"] = df_filtered[time_col].min().strftime("%Y-%m-%d")
    dataset_info["filtered_dataset_end"] = df_filtered[time_col].max().strftime("%Y-%m-%d")       
    dataset_info["filtered_dataset_max_duration_days"] = max_case_duration_days(df_filtered, time_col, case_col)
    dataset_info["train_cases"] = train_info[case_col].nunique()
    dataset_info["train_events"] = len(train_info)
    dataset_info["train_start"] = train_info[time_col].min().strftime("%Y-%m-%d")
    dataset_info["train_end"] = train_info[time_col].max().strftime("%Y-%m-%d")
    dataset_info["train_max_duration_days"] = max_case_duration_days(train_info, time_col, case_col, None)
    dataset_info["train_test_separation_time"] = pd.to_datetime(tt_st).strftime("%Y-%m-%d")
    #if test_split == "prefix":
    dataset_info["train_test_intersect_cases"] = tt_trunc[case_col].nunique()
    dataset_info["train_test_intersect_start"] = tt_trunc[time_col].min().strftime("%Y-%m-%d")
    dataset_info["train_test_intersect_end"] = tt_trunc[time_col].max().strftime("%Y-%m-%d")
    dataset_info["train_test_intersect_max_duration_days"] = max_case_duration_days(tt_trunc, time_col, case_col, None)
    dataset_info["test_cases"] = test_info[case_col].nunique()
    dataset_info["test_events"] = len(test_info)
    dataset_info["test_start"] = test_info[time_col].min().strftime("%Y-%m-%d")
    dataset_info["test_end"] = test_info[time_col].max().strftime("%Y-%m-%d")
    dataset_info["test_max_duration_days"] = max_case_duration_days(test_info, time_col, case_col, None)
    if val is not None:
        dataset_info["train_val_separation_time"] = pd.to_datetime(tv_st).strftime("%Y-%m-%d")
        dataset_info["val_cases"] = val_info[case_col].nunique()
        dataset_info["val_events"] = len(val_info)
        dataset_info["val_start"] = val_info[time_col].min().strftime("%Y-%m-%d")
        dataset_info["val_end"] = val_info[time_col].max().strftime("%Y-%m-%d")
        dataset_info["val_max_duration_days"] = max_case_duration_days(val_info, time_col, case_col, None)
    if val_split == "prefix" and tv_trunc is not None:
        dataset_info["train_val_intersect_cases"] = tv_trunc[case_col].nunique()
        dataset_info["train_val_intersect_start"] = tv_trunc[time_col].min().strftime("%Y-%m-%d")
        dataset_info["train_val_intersect_end"] = tv_trunc[time_col].max().strftime("%Y-%m-%d")
        dataset_info["train_val_intersect_max_duration_days"] = max_case_duration_days(tv_trunc, time_col, case_col, None)
    

    for j, v in dataset_info.items():
        print(f"{j}: {v}")    
        
    return dataset_info