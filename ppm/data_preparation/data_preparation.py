import pandas as pd
import numpy as np

from typing import Tuple, Optional


from skpm.event_logs import (
    BPI12,
    BPI15,
    BPI17,
    BPI19,
    BPI20PrepaidTravelCosts,
    BPI20TravelPermitData,
    BPI20RequestForPayment,
)

from skpm.feature_extraction import TimestampExtractor

from sklearn.preprocessing import StandardScaler

from ppm.datasets import ContinuousTraces
from ppm.datasets.utils import continuous
from ppm.datasets.event_logs import EventFeatures, EventLog, EventTargets

from torch.utils.data import DataLoader

from ppm.data_preparation.split import train_val_split_classic, nep_train_test_split

from ppm.data_preparation.dataset_info import dataset_statistics


EVENT_LOGS = {
    "BPI12": BPI12,
    "BPI15": BPI15,
    "BPI17": BPI17,
    "BPI19": BPI19,
    "BPI20PrepaidTravelCosts": BPI20PrepaidTravelCosts,
    "BPI20TravelPermitData": BPI20TravelPermitData,
    "BPI20RequestForPayment": BPI20RequestForPayment,
}

BENCHMARK_PARAMS = {
    "BPI12": {
        "start_date": None,
        "end_date": "2012-02",
        "max_days": 32.28,
        "test_size": 0.2,
    },
    "BPI15": {
        "start_date": "2010-10",
        "end_date": "2015-03",
        "max_days": 308.82,
        "test_size": 0.2,
    },
    "BPI17": {
        "start_date": None,
        "end_date": "2017-01",
        "max_days": 47.81,
        "test_size": 0.2,
    },
    "BPI19": {
        "start_date": "2018-01",
        "end_date": "2019-02",
        "max_days": 143.33,
        "test_size": 0.2,
    },
    "BPI20PrepaidTravelCosts": {
        "start_date": None,
        "end_date": "2019-01",
        "max_days": 114.26,
        "test_size": 0.2,
    },
    "BPI20TravelPermitData": {
        "start_date": None,
        "end_date": "2019-10",
        "max_days": 258.81,
        "test_size": 0.2,
    },
    "BPI20RequestForPayment": {
        "start_date": None,
        "end_date": "2018-12",
        "max_days": 28.86,
        "test_size": 0.2,
    },
}

NUMERICAL_FEATURES = [
    "accumulated_time",
    "day_of_month",
    "day_of_week",
    "day_of_year",
    "hour_of_day",
    "min_of_hour",
    "month_of_year",
    "sec_of_min",
    "secs_within_day",
    "week_of_year",
]


def filter_cases_starting_from(df: pd.DataFrame, start_date: str, case_col: str = "case:concept:name", time_col: str = "time:timestamp") -> pd.DataFrame:
    """
    Keep only cases that started on or after the given month.

    Args:
        df: Event log DataFrame.
        start_date: Year-month string (e.g., "2020-01"). If None, returns df unchanged.
        case_col: Column name for case identifier.
        time_col: Column name for timestamp.

    Returns:
        Filtered DataFrame containing only cases starting from the given month.
    """
    if start_date is None:
        return df
    case_starts = df.groupby(case_col)[time_col].min().reset_index()
    case_starts["date"] = case_starts[time_col].dt.to_period("M")
    cases_after = case_starts[case_starts["date"].astype(str) >= start_date][case_col].values
    return df[df[case_col].isin(cases_after)]


def filter_cases_ending_before(df: pd.DataFrame, end_date: str, case_col: str = "case:concept:name", time_col: str = "time:timestamp") -> pd.DataFrame:
    """
    Keep only cases that ended on or before the given month.

    Args:
        df: Event log DataFrame.
        end_date: Year-month string (e.g., "2020-12"). If None, returns df unchanged.
        case_col: Column name for case identifier.
        time_col: Column name for timestamp.

    Returns:
        Filtered DataFrame containing only cases ending before the given month.
    """
    if end_date is None:
        return df
    case_ends = df.groupby(case_col)[time_col].max().reset_index()
    case_ends["date"] = case_ends[time_col].dt.to_period("M")
    cases_before = case_ends[case_ends["date"].astype(str) <= end_date][case_col].values
    return df[df[case_col].isin(cases_before)]


def filter_cases_by_duration(df: pd.DataFrame, max_days: float, case_col: str = "case:concept:name", time_col: str = "time:timestamp") -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Filter cases to remove duration outliers and right-censored cases.

    This function applies two filters:
    1. Removes cases whose total duration exceeds max_days (outlier removal).
    2. Removes cases that started too late to be fully observable within the
       observation window (right-censoring bias removal). A case is considered
       potentially right-censored if it started after (max_timestamp - max_days),
       since we cannot know if it would have completed within max_days.

    Args:
        df: Event log DataFrame.
        max_days: Maximum allowed case duration in days.
        case_col: Column name for case identifier.
        time_col: Column name for timestamp.

    Returns:
        Tuple of (filtered DataFrame, latest allowed start timestamp).
    """
    agg = df.groupby(case_col)[time_col].agg(["min", "max"])
    agg["duration"] = (agg["max"] - agg["min"]).dt.total_seconds() / (24 * 60 * 60)

    cases_short = agg[np.isclose(agg["duration"], max_days) | (agg["duration"] < max_days)].index
    df = df[df[case_col].isin(cases_short)].reset_index(drop=True)

    latest_start = df[time_col].max() - pd.Timedelta(max_days, unit="D")
    agg_filtered = df.groupby(case_col)[time_col].min()
    cases_early = agg_filtered[agg_filtered <= latest_start].index
    df = df[df[case_col].isin(cases_early)].reset_index(drop=True)

    return df, latest_start


def prepare_data_with_prefix(
    df_orig: pd.DataFrame,
    log_name: str,
    val_split: str,
    val_size: float,
    lifecycle: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Extended prepare_data with prefix split and validation set support."""
    case_col = "case:concept:name"
    time_col = "time:timestamp"
    
    # If lifecycle is True, combine activity name with lifecycle transition
    if lifecycle:
        if "lifecycle:transition" in df_orig.columns:
            df_orig["concept:name"] = (
                df_orig["concept:name"] + "_" + df_orig["lifecycle:transition"]
            )
        else:
            print("Column 'lifecycle:transition' not found. Disabling lifecycle.")
            lifecycle = False
        
    cols_to_keep = [case_col, "concept:name", time_col]
    if "org:resource" in df_orig.columns:
        cols_to_keep.append("org:resource")
    if "case:AMOUNT_REQ" in df_orig.columns:
        cols_to_keep.append("case:AMOUNT_REQ")
    if "case:RequestedAmount" in df_orig.columns:
        cols_to_keep.append("case:RequestedAmount")
        
    df_orig = df_orig.loc[:, cols_to_keep].copy()
    df_orig[time_col] = pd.to_datetime(df_orig[time_col], utc=True)

    cases_to_keep = df_orig.groupby(case_col).size() > 2
    cases_to_keep = cases_to_keep[cases_to_keep].index
    df_orig = df_orig[df_orig[case_col].isin(cases_to_keep)]
    df_orig = df_orig.sort_values(by=[case_col, time_col])

    train_target_mask = None
    test_target_mask = None
    val_target_mask = None
    val = None

    params = BENCHMARK_PARAMS.get(log_name, {})
    df_filtered = filter_cases_starting_from(df_orig, params.get("start_date"))
    df_filtered = filter_cases_ending_before(df_filtered, params.get("end_date"))
    df_filtered = df_filtered.drop_duplicates()

    if params.get("max_days"):
        df_filtered, _ = filter_cases_by_duration(df_filtered, params["max_days"])

    test_size = params.get("test_size", 0.2)
    train, test, train_target_mask, test_target_mask, tt_trunc, tt_st = nep_train_test_split(df_filtered, test_size, case_col, time_col, None)

    if val_size > 0:
        if val_split == "prefix":
            train, val, train_target_mask, val_target_mask, tv_trunc, tv_st = nep_train_test_split(train, val_size, case_col, time_col, train_target_mask)
        else:
            train, val, train_target_mask, val_target_mask, tv_st= train_val_split_classic(train, train_target_mask, val_size)

    
    dataset_info = dataset_statistics(
        df_orig=df_orig,
        df_filtered=df_filtered, 
        train=train,
        test=test,
        val=val,
        val_split=val_split,
        tt_st=tt_st,
        tv_st=tv_st if val is not None else None,
        case_col=case_col,
        time_col=time_col,
        train_target_mask=train_target_mask,
        test_target_mask=test_target_mask,
        val_target_mask=val_target_mask,
        tt_trunc=tt_trunc, 
        tv_trunc=tv_trunc if val is not None and val_split == "prefix" else None,
        )
    
    ts = TimestampExtractor(
        case_features=["accumulated_time", "remaining_time"],
        event_features="all",
        time_unit="d",
    )
    train[ts.get_feature_names_out()] = ts.fit_transform(train)
    test[ts.get_feature_names_out()] = ts.transform(test)

    train = train.drop(columns=[time_col])
    test = test.drop(columns=[time_col])

    rename_map = {case_col: "case_id", "concept:name": "activity"}
    if "org:resource" in train.columns:
        rename_map["org:resource"] = "resource"
        train["org:resource"] = train["org:resource"].fillna("UNKNOWN")
        test["org:resource"] = test["org:resource"].fillna("UNKNOWN")
    if "case:AMOUNT_REQ" in train.columns:
        rename_map["case:AMOUNT_REQ"] = "amount"
        train["case:AMOUNT_REQ"] = pd.to_numeric(train["case:AMOUNT_REQ"], errors="coerce").fillna(0).astype(float)
        test["case:AMOUNT_REQ"] = pd.to_numeric(test["case:AMOUNT_REQ"], errors="coerce").fillna(0).astype(float)
    if "case:RequestedAmount" in train.columns:
        rename_map["case:RequestedAmount"] = "amount"
        train["case:RequestedAmount"] = pd.to_numeric(train["case:RequestedAmount"], errors="coerce").fillna(0).astype(float)
        test["case:RequestedAmount"] = pd.to_numeric(test["case:RequestedAmount"], errors="coerce").fillna(0).astype(float)
    train = train.rename(columns=rename_map)
    test = test.rename(columns=rename_map)

    sc = StandardScaler()
    columns = NUMERICAL_FEATURES + ["remaining_time"]
    if "amount" in train.columns:
        columns = columns + ["amount"]
    train.loc[:, columns] = sc.fit_transform(train[columns])
    test.loc[:, columns] = sc.transform(test[columns])

    if val is not None:
        val[ts.get_feature_names_out()] = ts.transform(val)
        val = val.drop(columns=[time_col])
        if "org:resource" in val.columns:
            val["org:resource"] = val["org:resource"].fillna("UNKNOWN")
        if "case:AMOUNT_REQ" in val.columns:
            val["case:AMOUNT_REQ"] = pd.to_numeric(val["case:AMOUNT_REQ"], errors="coerce").fillna(0).astype(float)
        if "case:RequestedAmount" in val.columns:
            val["case:RequestedAmount"] = pd.to_numeric(val["case:RequestedAmount"], errors="coerce").fillna(0).astype(float)
        val = val.rename(columns=rename_map)
        val.loc[:, columns] = sc.transform(val[columns])

    return train, test, val, train_target_mask, test_target_mask, val_target_mask, dataset_info



def charge_loaders(training_config: dict):
    log_name = training_config["log"]
    log = EVENT_LOGS[log_name]()
    
    val_split = training_config.get("val_split", "classic")
    val_size = training_config.get("val_size", 0.0)
    lifecycle = training_config.get("lifecycle", False)

    train, test, val, train_mask, test_mask, val_mask, dataset_info = prepare_data_with_prefix(
        df_orig=log.dataframe,
        log_name=log_name,
        val_split=val_split,
        val_size=val_size,
        lifecycle=lifecycle,
    )

    event_features = EventFeatures(
        categorical=training_config["categorical_features"],
        numerical=[] if training_config["continuous_features"] is None else training_config["continuous_features"],
    )

    event_targets = EventTargets(
        categorical=training_config["categorical_targets"],
        numerical=training_config["continuous_targets"],
    )



    # Remark: targets are computed inside EventLog
    train_log = EventLog(
        dataframe=train,
        case_id="case_id",
        features=event_features,
        targets=event_targets,
        train_split=True,
        name=log_name,
        target_mask=train_mask,
    )

    test_log = EventLog(
        dataframe=test,
        case_id="case_id",
        features=event_features,
        targets=event_targets,
        train_split=False,
        name=log_name,
        vocabs=train_log.get_vocabs(),
        target_mask=test_mask,
    )

    val_log = None
    if val is not None:
        val_log = EventLog(
            dataframe=val,
            case_id="case_id",
            features=event_features,
            targets=event_targets,
            train_split=False,
            name=log_name,
            vocabs=train_log.get_vocabs(),
            target_mask=val_mask,
        )

    # Always keep dataset tensors on CPU; DataLoader workers must not touch CUDA.
    # Data movement to GPU is handled later by the training loop/prefetcher.
    dataset_device = "cpu"

    train_dataset = ContinuousTraces(log=train_log, refresh_cache=True, device=dataset_device)
    test_dataset = ContinuousTraces(log=test_log, refresh_cache=True, device=dataset_device)

    val_dataset = None
    if val_log is not None:
        val_dataset = ContinuousTraces(log=val_log, refresh_cache=True, device=dataset_device)

    num_workers = training_config.get("num_workers", 8)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,    #Anonymous Author set to True, was False
        collate_fn=continuous,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        collate_fn=continuous,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config["batch_size"],
            shuffle=False,
            collate_fn=continuous,
        )

    return train_log, train_loader, test_loader, val_loader, dataset_info

