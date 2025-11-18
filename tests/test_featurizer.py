from .conftest import (
    CDR_DATA,
    MOBILE_DATA_USAGE_DATA,
    MOBILE_MONEY_TRANSACTION_DATA,
    RECHARGE_DATA,
)
import pandas as pd
import pytest
from cider.featurizer.dependencies import (
    filter_to_datetime,
    get_spammers_from_cdr_data,
    get_outlier_days_from_cdr_data,
    get_standard_diagnostic_statistics,
)


class TestFeaturizerInference:

    @pytest.mark.parametrize(
        "dataset",
        [
            CDR_DATA,
            MOBILE_DATA_USAGE_DATA,
            MOBILE_MONEY_TRANSACTION_DATA,
            RECHARGE_DATA,
        ],
    )
    def test_filter_to_datetime(self, dataset):
        df = pd.DataFrame(dataset)
        df_with_duplicates = pd.concat([df, df.loc[:2]], ignore_index=True)
        filtered_data = filter_to_datetime(
            df_with_duplicates,
            filter_start_date=pd.to_datetime("2023-01-02"),
            filter_end_date=pd.to_datetime("2023-01-03"),
        )

        assert all(
            (filtered_data["timestamp"] >= pd.to_datetime("2023-01-02 00:00:00"))
            & (filtered_data["timestamp"] <= pd.to_datetime("2023-01-03 23:59:59"))
        )
        assert len(filtered_data) == 2

    def test_get_spammers_from_cdr_data(self):
        cdr = pd.DataFrame(CDR_DATA)
        # Add spammer data
        spammer_data = {
            "caller_id": ["spammer_1"] * 300,
            "recipient_id": ["recipient_spam"] * 300,
            "caller_antenna_id": ["antenna_spam"] * 300,
            "timestamp": pd.date_range(
                start="2023-01-01 00:00:00", periods=300, freq="5min"
            ),
            "duration": [60] * 300,
            "transaction_type": ["call"] * 300,
            "transaction_scope": ["domestic"] * 300,
        }
        spammer_cdr = pd.DataFrame(spammer_data)
        cdr_with_spammer = pd.concat([cdr, spammer_cdr], ignore_index=True)

        spammer_ids = get_spammers_from_cdr_data(
            cdr_with_spammer, threshold_of_calls_per_day=100
        )

        assert spammer_ids == ["spammer_1"]

    def test_get_outlier_days_from_cdr_data(self):
        cdr = pd.DataFrame(CDR_DATA)
        # Add outlier day data
        outlier_data = {
            "caller_id": ["caller_outlier"] * 1000,
            "recipient_id": ["recipient_outlier"] * 1000,
            "caller_antenna_id": ["antenna_outlier"] * 1000,
            "timestamp": pd.date_range(
                start="2023-01-10 00:00:01", periods=1000, freq="s"
            ),
            "duration": [60] * 1000,
            "transaction_type": ["call"] * 1000,
            "transaction_scope": ["domestic"] * 1000,
        }
        outlier_cdr = pd.DataFrame(outlier_data)
        cdr_with_outlier = pd.concat([cdr, outlier_cdr], ignore_index=True)

        outlier_days = get_outlier_days_from_cdr_data(
            cdr_with_outlier, zscore_threshold=1.0
        )

        assert pd.to_datetime("2023-01-10").date() in outlier_days
        assert len(outlier_days) == 1

    @pytest.mark.parametrize(
        "data",
        [
            CDR_DATA,
            MOBILE_DATA_USAGE_DATA,
            MOBILE_MONEY_TRANSACTION_DATA,
            RECHARGE_DATA,
        ],
    )
    def test_get_standard_diagnostic_statistics(self, data):
        df = pd.DataFrame(data)
        stats = get_standard_diagnostic_statistics(df)

        assert stats.total_transactions == len(df)
        assert stats.num_unique_callers == df["caller_id"].nunique()
        assert stats.num_days == df["timestamp"].dt.date.nunique()
        if "recipient_id" in df.columns:
            assert stats.num_unique_recipients == df["recipient_id"].nunique()
        else:
            assert stats.num_unique_recipients == 0
