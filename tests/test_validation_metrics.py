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

import pytest
import pandas as pd
from cider.validation_metrics.dependencies import (
    calculate_weighted_spearmanr,
    calculate_weighted_pearsonr,
    convert_threshold_to_percentile,
    calculate_metrics_binary_valued_consumption,
    calculate_utility,
)
from cider.validation_metrics.schemas import HouseholdConsumptionData, ConsumptionColumn
from conftest import HOUSEHOLD_CONSUMPTION_DATA


class TestValidationMetricsDependencies:

    household_consumption_data = pd.DataFrame(HOUSEHOLD_CONSUMPTION_DATA)

    def test_missing_columns_raise_errors(self):
        for col in HouseholdConsumptionData.model_fields.keys():
            household_data_no_cols = self.household_consumption_data.drop(columns=[col])
            with pytest.raises(ValueError):
                convert_threshold_to_percentile(50.0, household_data_no_cols)
            with pytest.raises(ValueError):
                calculate_weighted_spearmanr(household_data_no_cols)
            with pytest.raises(ValueError):
                calculate_weighted_pearsonr(household_data_no_cols)
            with pytest.raises(ValueError):
                calculate_metrics_binary_valued_consumption(
                    household_data_no_cols, 50.0, 50.0
                )
            with pytest.raises(ValueError):
                calculate_utility(
                    household_data_no_cols, 50.0, ConsumptionColumn.GROUNTRUTH, 1000
                )

    @pytest.mark.parametrize(
        "threshold,expected_percentile",
        [
            (5.0, 79.69),
            (2.0, 30.52),
            (1.0, 27.35),
        ],
    )
    def test_convert_threshold_to_percentile(self, threshold, expected_percentile):
        percentile = convert_threshold_to_percentile(
            threshold, self.household_consumption_data
        )
        assert pytest.approx(percentile, 1e-3) == expected_percentile

    @pytest.mark.parametrize(
        "significant_digits,expected_spearmanr",
        [
            (2, 0.97),
            (3, 0.967),
            (4, 0.9675),
        ],
    )
    def test_calculate_weighted_spearmanr(self, significant_digits, expected_spearmanr):
        spearmanr = calculate_weighted_spearmanr(
            self.household_consumption_data, significant_digits=significant_digits
        )
        assert (
            pytest.approx(spearmanr, 10 ** (-significant_digits)) == expected_spearmanr
        )

    @pytest.mark.parametrize(
        "significant_digits,expected_pearsonr",
        [
            (2, 0.93),
            (3, 0.93),
            (4, 0.93),
        ],
    )
    def test_calculate_weighted_pearsonr(self, significant_digits, expected_pearsonr):
        pearsonr = calculate_weighted_pearsonr(
            self.household_consumption_data, significant_digits=significant_digits
        )
        assert pytest.approx(pearsonr, 10 ** (-significant_digits)) == expected_pearsonr

    @pytest.mark.parametrize(
        "groundtruth_threshold_percentile,proxy_threshold_percentile,expected_accuracy,expected_precision,expected_recall,expected_tpr,expected_fpr,expected_auc,expected_roc_curve",
        [
            (
                25,
                25,
                0.625,
                0.786,
                0.647,
                0.647,
                0.429,
                0.3,
                ([0.0, 0.733, 1.0], [0.0, 0.33, 1.0]),
            ),
            (
                25,
                50,
                0.625,
                0.667,
                0.5,
                0.5,
                0.25,
                0.467,
                ([0.0, 0.4, 1.0], [0.0, 0.33, 1.0]),
            ),
            (
                50,
                25,
                0.4583,
                0.786,
                0.524,
                0.524,
                1.0,
                0.115,
                ([0.0, 1.0, 1.0], [0.0, 0.231, 1.0]),
            ),
        ],
    )
    def test_calculate_metrics_binary_valued_consumption(
        self,
        groundtruth_threshold_percentile,
        proxy_threshold_percentile,
        expected_accuracy,
        expected_precision,
        expected_recall,
        expected_tpr,
        expected_fpr,
        expected_auc,
        expected_roc_curve,
    ):
        results = calculate_metrics_binary_valued_consumption(
            self.household_consumption_data,
            groundtruth_threshold_percentile,
            proxy_threshold_percentile,
        )
        for k, v in results.items():
            print(f"{k}: {v}")
        assert pytest.approx(results["accuracy"], 1e-2) == expected_accuracy
        assert pytest.approx(results["precision"], 1e-2) == expected_precision
        assert pytest.approx(results["recall"], 1e-2) == expected_recall
        assert pytest.approx(results["true_positive_rate"], 1e-2) == expected_tpr
        assert pytest.approx(results["false_positive_rate"], 1e-2) == expected_fpr
        assert pytest.approx(results["auc"], 1e-2) == expected_auc
        roc_curve_fpr, roc_curve_tpr, _ = results["roc_curve"].to_numpy()[0]
        assert pytest.approx(roc_curve_fpr, 1e-2) == expected_roc_curve[0]
        assert pytest.approx(roc_curve_tpr, 1e-2) == expected_roc_curve[1]

    @pytest.mark.parametrize(
        "consumption_column,threshold_percentile,expected_utility",
        [
            (ConsumptionColumn.GROUNTRUTH, 20.0, -0.01712),
            (ConsumptionColumn.PROXY, 30.0, -0.00614),
        ],
    )
    def test_calculate_utility(
        self, consumption_column, threshold_percentile, expected_utility
    ):
        utility = calculate_utility(
            self.household_consumption_data,
            threshold_percentile=threshold_percentile,
            consumption_column=consumption_column,
            cash_transfer_amount=1000,
        )
        assert pytest.approx(utility, 1e-3) == expected_utility
