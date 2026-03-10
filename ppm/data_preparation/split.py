import pandas as pd
import numpy as np
from typing import Tuple, Optional



def train_val_split_classic(df: pd.DataFrame, train_target_mask: np.ndarray, val_size: float, case_col: str = "case:concept:name", time_col: str = "time:timestamp") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split training data into train and validation sets by case.

    Selects the most recent cases (by earliest event timestamp) for validation.
    This ensures temporal ordering where validation cases started after training cases.

    If a train_target_mask (aligned with df rows) is provided, it is split
    alongside the DataFrame rows so that:
      - the returned train_target_mask corresponds to the remaining training rows
      - the returned val_target_mask corresponds to the validation rows

    Args:
        df: Training DataFrame to split.
        train_target_mask: Optional boolean mask aligned with df rows. If None, no masks are returned.
        val_size: Fraction of cases to use for validation (e.g., 0.2 for 20%).
        case_col: Column name for case identifier.
        time_col: Column name for timestamp.

    Returns:
        If train_target_mask is None:
            (train_df, val_df)
        Otherwise:
            (train_df, val_df, train_target_mask_split, val_target_mask_split)
    """
    n_cases = df[case_col].nunique()
    val_case_ids = (
        df.groupby(case_col)[time_col]
        .min()
        .sort_values(ascending=False)
        .head(int(val_size * n_cases))
        .index
    )

    # Boolean index over original df indicating which rows go to validation
    val_row_mask = df[case_col].isin(val_case_ids).values
    train_row_mask = ~val_row_mask

    # Split the frames (order preserved), then reset indices
    val = df[val_row_mask].reset_index(drop=True)
    train = df[train_row_mask].reset_index(drop=True)
    
    # for information only: ealriest event in val set
    separation_time = val[time_col].min()

    if train_target_mask is not None:
        # Ensure numpy array and correct shape
        train_target_mask = np.asarray(train_target_mask).reshape(-1)
        if train_target_mask.shape[0] != df.shape[0]:
            raise ValueError("train_target_mask must be aligned with df rows in train_val_split_classic")
        # Partition the mask in the same order as the data split
        new_train_mask = train_target_mask[train_row_mask]
        val_target_mask = train_target_mask[val_row_mask]
        return train, val, new_train_mask, val_target_mask, separation_time
    else:
        return train, val, separation_time



def nep_train_test_split(
    df: pd.DataFrame, split_size: float, case_col: str = "case:concept:name", time_col: str = "time:timestamp",
    train_target_mask_orig: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Perform a temporal prefix-based train/test split with target masking.

    This implements the prefix-based splitting strategy where:
    - Train set contains all events up to and including the temporal separation time.
    - Test set contains all events from cases that end after the separation time.
    - Target masks indicate which events should be excluded from loss computation.

    For a case with events A,B,C,<cutoff>,D,E:
    - Train gets A,B,C with C's target masked (since D is unknown at cutoff time).
    - Test gets A,B,C,D,E with A,B masked (already known) masked.
    
    This function can also be used to split a training set into train/val sets.
    In that case, the original train_target_mask (aligned with df rows) is taken into account
    to ensure that masked events remain masked in the resulting splits.

    Args:
        df: Event log DataFrame sorted by case and timestamp.
        split_size: Fraction of timestamps to place after the cutoff.
        case_col: Column name for case identifier.
        time_col: Column name for timestamp.
        train_target_mask_orig: Optional boolean mask aligned with df rows. 

    Returns:
        Tuple of (train, test, train_target_mask, test_target_mask).
        Masks are boolean arrays where True indicates the target should be masked.
    """
    if split_size <= 0.0:
        return df, None, None, None
    
    if df[time_col].dt.tz is None:
        df[time_col] = df[time_col].dt.tz_localize("UTC")
        
    # sort by case and time to ensure correct order
    df_sorted = df.sort_values(by=[case_col, time_col])
    if train_target_mask_orig is not None:
        train_target_mask_orig = train_target_mask_orig[df_sorted.index.to_numpy()]
    df = df_sorted

    # calculate separation time based on timestamps
    sorted_timestamps = np.sort(df[time_col].values)
    separation_time = sorted_timestamps[int(len(sorted_timestamps) * (1 - split_size))]
    
    # train set: all events up to and including separation time
    train = df[df[time_col] <= pd.to_datetime(separation_time, utc=True)].reset_index(drop=True)
    
    # test set: all events of cases that end after separation time
    case_max_ts = df.groupby(case_col)[time_col].max()
    test_case_nrs = set(case_max_ts[case_max_ts > pd.to_datetime(separation_time, utc=True)].index.array)
    test = df[df[case_col].isin(test_case_nrs)].reset_index(drop=True)
    
    # determine truncated cases to calculate masks
    case_min_ts = df.groupby(case_col)[time_col].min()
    cases_starting_before_separation_time = set(case_min_ts[case_min_ts <= pd.to_datetime(separation_time, utc=True)].index)
    truncated_cases = cases_starting_before_separation_time.intersection(test_case_nrs)
    trunc = df[df[case_col].isin(truncated_cases)].reset_index(drop=True)   #for information about datasets only

    # no prediction for last event in train of truncated cases (its target is unknown since the case continues until after separation time)
    last_idx = train.groupby(case_col, sort=False).tail(1).index
    train_is_last = train.index.isin(last_idx)

    train_is_truncated = train[case_col].isin(truncated_cases)
    train_target_mask = (train_is_last & train_is_truncated).values
    
    # no prediction for prefixes in test set whose targets are before or on the separation time
    test = test.sort_values([case_col, time_col]).reset_index(drop=True)
    test_next_ts = test.groupby(case_col)[time_col].shift(-1)  #last event is per definition after separation time, so the shift will cause no problems for it
    test_target_mask = (test_next_ts <= pd.to_datetime(separation_time, utc=True)).to_numpy()
    
    # if the test set refers to the validation set, we need to integrate the original train_target_mask
    if train_target_mask_orig is not None:
        # for train set (masked prefixes in original train set can end up in new train set, and need to be masked there as well)
        train_target_mask_orig_train = train_target_mask_orig[df[time_col] <= pd.to_datetime(separation_time, utc=True)]
        train_target_mask = train_target_mask | train_target_mask_orig_train
        # for val set 
        train_target_mask_orig_val = train_target_mask_orig[df[case_col].isin(test_case_nrs)]
        test_target_mask = test_target_mask | train_target_mask_orig_val

    return train, test, train_target_mask, test_target_mask, trunc, separation_time

