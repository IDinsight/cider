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

from cider.utils import _validate_dataframe
from .dependencies import (
    calculate_metrics_binary_valued_consumption,
    where_is_false_positive_rate_nonmonotonic,
)
from .schemas import HouseholdConsumptionData
import pandas as pd
import numpy as np


def compute_auc_roc_with_percentile_grid(
    data: pd.DataFrame,
    num_grid_points: int = 99,
) -> pd.DataFrame:
    """
    Compute AUC-ROC across a grid of percentiles.

    Args:
        data (pd.DataFrame): Data containing 'groundtruth_consumption', 'proxy_consumption', and 'weight' values.
        num_grid_points (int): Number of grid points to compute AUC-ROC.

    Returns:
        pd.DataFrame: DataFrame containing percentiles and corresponding false positive rates, false negative rates, and AUC values.
    """
    # Validate that input data has the required columns
    _validate_dataframe(data, required_schema=HouseholdConsumptionData)

    # Create percentile grid
    percentiles = np.linspace(1, 99, num_grid_points)[::-1]

    results_per_grid = [
        calculate_metrics_binary_valued_consumption(data, p, p) for p in percentiles
    ]
    true_positive_rates = [
        result.true_positive_rate.values[0] for result in results_per_grid
    ]
    false_positive_rates = [
        result.false_positive_rate.values[0] for result in results_per_grid
    ]
    auc_values = [result.auc.values[0] for result in results_per_grid]

    nonmonotonic_indices = where_is_false_positive_rate_nonmonotonic(
        np.array(false_positive_rates)
    )
    if len(nonmonotonic_indices) > 0:
        false_positive_rates = [
            false_positive_rates[i]
            for i in range(len(false_positive_rates))
            if i not in nonmonotonic_indices
        ]
        true_positive_rates = [
            true_positive_rates[i]
            for i in range(len(true_positive_rates))
            if i not in nonmonotonic_indices
        ]
        percentiles = [
            percentiles[i]
            for i in range(len(percentiles))
            if i not in nonmonotonic_indices
        ]
        auc_values = [
            auc_values[i]
            for i in range(len(auc_values))
            if i not in nonmonotonic_indices
        ]

    results_df = pd.DataFrame(
        {
            "percentile": percentiles,
            "true_positive_rate": true_positive_rates,
            "false_positive_rate": false_positive_rates,
            "auc": auc_values,
        }
    )
    return results_df


# def compute_utility_grid()
