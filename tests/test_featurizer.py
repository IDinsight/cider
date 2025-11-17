from .conftest import CDR_DATA
import pandas as pd

from cider.featurizer.inference import filter_to_datetime, get_spammers_from_cdr_data


class TestFeaturizerInference:

    def test_filter_to_datetime(self):
        cdr = pd.DataFrame(CDR_DATA)
        cdr_with_duplicates = pd.concat([cdr, cdr.loc[:2]], ignore_index=True)
        filtered_cdr = filter_to_datetime(
            cdr_with_duplicates,
            filter_start_date=pd.to_datetime("2023-01-02"),
            filter_end_date=pd.to_datetime("2023-01-03"),
        )

        assert all(
            (filtered_cdr["timestamp"] >= pd.to_datetime("2023-01-02 00:00:00"))
            & (filtered_cdr["timestamp"] <= pd.to_datetime("2023-01-03 23:59:59"))
        )
        assert len(filtered_cdr) == 2

    def test_remove_spammers_from_cdr_data(self):
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
