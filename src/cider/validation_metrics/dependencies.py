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
import numpy as np

from cider.utils import _validate_dataframe
from .schemas import HouseholdConsumptionData, ConsumptionColumn
from scipy.stats import rankdata
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def convert_threshold_to_percentile(
    threshold: float | list[float],
    data: pd.DataFrame,
    consumption_column: ConsumptionColumn = ConsumptionColumn.GROUNTRUTH,
) -> float:
    """Convert a threshold value to its corresponding percentile in the data.

    Args:
        threshold (float | list[float]): The threshold value(s) to convert.
        data (pd.DataFrame): Data containing 'groundtruth_consumption', 'proxy_consumption', and 'weight' values.
        consumption_column (ConsumptionColumn): The column name for consumption values in the data.

    Returns:
        float: The percentile corresponding to the threshold value.
    """
    # Validate that input data has the required columns
    _validate_dataframe(data, required_schema=HouseholdConsumptionData)

    # Calculate and return the percentile
    consumption_data = data[consumption_column.value].to_numpy()
    weights = data["weight"].to_numpy()
    sorted_indices = np.argsort(consumption_data)
    sorted_consumption = consumption_data[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute the cumulative weights
    cumulative_weights = np.cumsum(sorted_weights)
    normalized_cumulative_weights = (cumulative_weights / cumulative_weights[-1]) * 100

    # Convert threshold to percentile
    percentile = np.atleast_1d(threshold)
    percentile = np.interp(
        percentile, sorted_consumption, normalized_cumulative_weights
    )

    return percentile if len(percentile) > 1 else percentile[0]


def calculate_weighted_spearmanr(
    data: pd.DataFrame, significant_digits: int = 2
) -> float:
    """
    Calculate the weighted Spearman correlation
    Args:
        data (pd.DataFrame): DataFrame containing 'groundtruth_consumption', 'proxy_consumption', and 'weight' columns.
    Returns:
        float: Weighted Spearman correlation coefficient.
    """
    # Validate that input data has the required columns
    _validate_dataframe(data, required_schema=HouseholdConsumptionData)

    # Rank the groundtruth and proxy consumption values
    rank_groundtruth = rankdata(data["groundtruth_consumption"], method="average")
    rank_proxy = rankdata(data["proxy_consumption"], method="average")

    # Compute weighted Spearman correlation
    normalized_groundtruth = rank_groundtruth - np.mean(
        data["weight"] * rank_groundtruth
    )
    normalized_proxy = rank_proxy - np.mean(data["weight"] * rank_proxy)

    numerator = np.sum(data["weight"] * normalized_groundtruth * normalized_proxy)
    denominator = np.sqrt(
        np.sum(data["weight"] * normalized_groundtruth**2)
        * np.sum(data["weight"] * normalized_proxy**2)
    )
    return round(numerator / denominator, significant_digits)


def calculate_weighted_pearsonr(
    data: pd.DataFrame, significant_digits: int = 2
) -> float:
    """
    Calculate the weighted Pearson correlation
    Args:
        data (pd.DataFrame): DataFrame containing 'groundtruth_consumption', 'proxy_consumption', and 'weight' columns.
    Returns:
        float: Weighted Pearson correlation coefficient.
    """
    # Validate that input data has the required columns
    _validate_dataframe(data, required_schema=HouseholdConsumptionData)

    # Compute weighted Pearson correlation
    covariance_matrix = np.cov(
        data["groundtruth_consumption"],
        data["proxy_consumption"],
        aweights=data["weight"],
    )

    pearsons_r = covariance_matrix[0, 1] / np.sqrt(
        covariance_matrix[0, 0] * covariance_matrix[1, 1]
    )
    return round(pearsons_r, significant_digits)


def calculate_metrics_binary_valued_consumption(
    data: pd.DataFrame,
    groundtruth_threshold_percentile: float,
    proxy_threshold_percentile: float,
) -> pd.DataFrame:
    """
    Calculate AUC curves for binary-valued (below / above threshold) consumption data at specified thresholds.

    Args:
        data (pd.DataFrame): DataFrame containing 'groundtruth_consumption', 'proxy_consumption' and 'weight' columns.
        groundtruth_threshold_percentile (float): Percentile threshold to use for groundtruth consumption calculation.
        proxy_threshold_percentile (float): Percentile threshold to use for proxy consumption calculation.
    Returns:
        pd.DataFrame: DataFrame containing binary metrics.
    """
    # Validate that input data has the required columns
    _validate_dataframe(data, required_schema=HouseholdConsumptionData)

    # Validate threshold values are correct
    if (
        not groundtruth_threshold_percentile > 0.0
        and groundtruth_threshold_percentile < 100
    ):
        raise ValueError("groundtruth_threshold_percentile must be between 0 and 100")
    if not proxy_threshold_percentile > 0.0 and proxy_threshold_percentile < 100:
        raise ValueError("proxy_threshold_percentile must be between 0 and 100")

    # Binarize consumption values based on thresholds
    groundtruth_threshold_value = np.percentile(
        data["groundtruth_consumption"], groundtruth_threshold_percentile
    )
    proxy_threshold_value = np.percentile(
        data["proxy_consumption"], proxy_threshold_percentile
    )

    groundtruth_binary = (
        data["groundtruth_consumption"] <= groundtruth_threshold_value
    ).astype(int)
    proxy_binary = (data["proxy_consumption"] <= proxy_threshold_value).astype(int)

    # Calculate metrics
    true_pos, true_neg, false_pos, false_neg = confusion_matrix(
        groundtruth_binary, proxy_binary, sample_weight=data["weight"]
    ).ravel()

    # Calculate rates
    results = {
        "accuracy": (true_pos + true_neg)
        / (true_pos + true_neg + false_pos + false_neg),
        "precision": true_pos / (true_pos + false_pos),
        "recall": true_pos / (true_pos + false_neg),
        "true_positive_rate": true_pos / (true_pos + false_neg),
        "false_positive_rate": false_pos / (false_pos + true_neg),
        "auc": roc_auc_score(
            groundtruth_binary, -proxy_binary, sample_weight=data["weight"]
        ),
        "roc_curve": roc_curve(
            groundtruth_binary, -proxy_binary, sample_weight=data["weight"]
        ),
    }

    return pd.DataFrame([results])


def calculate_utility(
    data: pd.DataFrame,
    threshold_percentile: float,
    consumption_column: ConsumptionColumn,
    cash_transfer_amount: float,
    constant_relative_risk_aversion: float = 3.0,
) -> float:
    """
    Computes the constant relative risk-aversion (CRRA) [Hanna & Olken (2018)] utility when P% of the population is
    targeted and they receive transfers of size 'transfer_size'

    Args:
        data (pd.DataFrame): DataFrame containing 'groundtruth_consumption', 'proxy_consumption' and 'weight'
        threshold_percentile (float): Percentile threshold to use for consumption calculation.
        consumption_column (ConsumptionColumn): Enum indicating which consumption column to use.
        cash_transfer_amount (float): Amount of cash transfer given to targeted households.
        constant_relative_risk_aversion (float): Coefficient of relative risk aversion (CRRA) utility function.

    Returns:
        float: The utility obtained by targeting the specified percentile of the population with the speciied cash transfer amount.
    """

    # Validate that input data has the required columns
    _validate_dataframe(data, required_schema=HouseholdConsumptionData)

    # Validate threshold values are correct
    if not threshold_percentile > 0.0 and threshold_percentile < 100:
        raise ValueError("threshold_percentile must be between 0 and 100")

    # Compute utility
    threshold_value = np.percentile(
        data[consumption_column.value], threshold_percentile
    )
    is_cash_transferred = (data[consumption_column.value] < threshold_value).astype(
        float
    )
    benefits = is_cash_transferred * data.weight * cash_transfer_amount
    utility = (
        (data[consumption_column.value] + benefits)
        ** (1 - constant_relative_risk_aversion)
    ) / (1 - constant_relative_risk_aversion)
    return (utility * data.weight).sum() / data.weight.sum()
