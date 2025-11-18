# Copyright ©2022-2023. The Regents of the University of California
# (Regents). All Rights Reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the
# distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pandas as pd
from datetime import datetime

from .schemas import DataDiagnosticStatistics


def filter_to_datetime(
    df: pd.DataFrame, filter_start_date: datetime, filter_end_date: datetime
) -> pd.DataFrame:
    """
    Filter dataframe to a specific datetime range.

    Args:
        df: pandas dataframe
        filter_start_date: start date to filter data
        filter_end_date: end date to filter data

    Returns:
        df: pandas dataframe
    """
    # Deduplicate entries
    df = df.drop_duplicates()

    # Filter by date range
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[
        (df["timestamp"] >= filter_start_date) & (df["timestamp"] <= filter_end_date)
    ]
    return df


def get_spammers_from_cdr_data(
    cdr_data: pd.DataFrame, threshold_of_calls_per_day: float = 100
) -> list[str]:
    """
    Remove spammers from CDR data based on a threshold of calls per day.
    Args:
        cdr_data: pandas dataframe with CDR data
        threshold_of_calls_per_day: threshold of calls per day to identify spammers
    Returns:
        spammers: list of caller IDs identified as spammers
    """
    # Extract day from timestamp
    cdr_data["day"] = cdr_data["timestamp"].dt.date

    # Get number of transactions per day per transaction type and per caller
    grouped_cdr_data = (
        (cdr_data.groupby(["caller_id", "transaction_type"]))
        .apply(
            lambda x: pd.Series(
                {
                    "caller_id": x.caller_id.iloc[0],
                    "count_transactions": x.shape[0],
                    "active_days": x["day"].nunique(),
                }
            )
        )
        .reset_index(drop=True)
    )

    grouped_cdr_data["avg_calls_per_day"] = (
        grouped_cdr_data["count_transactions"] / grouped_cdr_data["active_days"]
    )

    # Filter out callers with avg calls per day greater than threshold
    spammer_caller_ids = (
        grouped_cdr_data[
            grouped_cdr_data["avg_calls_per_day"] > threshold_of_calls_per_day
        ]
        .caller_id.unique()
        .tolist()
    )

    return spammer_caller_ids


def get_outlier_days_from_cdr_data(
    cdr_data: pd.DataFrame, zscore_threshold: float = 2.0
) -> list[str]:
    """
    Remove outlier days from CDR data based on z-score of daily transaction counts.

    Outlier days are those days where the number of transactions is beyond the
    specified z-score threshold from the mean number of daily transactions for each
    transaction type.

    Args:
        cdr_data: pandas dataframe with CDR data
        zscore_threshold: z-score threshold to identify outlier days
    Returns:
        cdr_data: pandas dataframe with outlier days removed
    """
    # Add day column
    cdr_data["day"] = cdr_data["timestamp"].dt.date

    # Group data by caller_id and day to get daily transaction counts
    daily_counts = cdr_data.groupby(["day", "transaction_type"], as_index=False).apply(
        lambda x: x.shape[0]
    )
    daily_counts = daily_counts.rename(columns={None: "daily_count"})

    per_transaction_mean = daily_counts.groupby(
        "transaction_type", as_index=False
    ).daily_count.mean()
    per_transaction_std = daily_counts.groupby(
        "transaction_type", as_index=False
    ).daily_count.std()

    bottom_thresholds = per_transaction_mean.copy()
    top_thresholds = per_transaction_mean.copy()

    bottom_thresholds.daily_count = per_transaction_mean.daily_count - (
        zscore_threshold * per_transaction_std.daily_count
    )
    top_thresholds.daily_count = per_transaction_mean.daily_count + (
        zscore_threshold * per_transaction_std.daily_count
    )

    # Map thresholds to each row's transaction_type
    daily_counts["bottom_threshold"] = daily_counts["transaction_type"].map(
        bottom_thresholds.set_index("transaction_type")["daily_count"]
    )
    daily_counts["top_threshold"] = daily_counts["transaction_type"].map(
        top_thresholds.set_index("transaction_type")["daily_count"]
    )

    # Get outlier days
    outlier_days = daily_counts[
        (daily_counts["daily_count"] < daily_counts["bottom_threshold"])
        | (daily_counts["daily_count"] > daily_counts["top_threshold"])
    ]["day"]

    return outlier_days.unique().tolist()


def get_standard_diagnostic_statistics(df: pd.DataFrame) -> DataDiagnosticStatistics:
    """
    Get standrd diagnostic statistics for CDR, recharge, mobile money and mobile phone data.

    Args:
        df: pandas dataframe
    Returns:
        statistics: DataDiagnosticStatistics object with diagnostic statistics
    """
    statistics = {
        "total_transactions": df.shape[0],
        "num_unique_callers": df["caller_id"].nunique(),
        "num_unique_recipients": (
            df["recipient_id"].nunique() if "recipient_id" in df.columns else 0
        ),
        "num_days": df["timestamp"].dt.date.nunique(),
    }
    return DataDiagnosticStatistics.model_validate(statistics)
