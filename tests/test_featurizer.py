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


from conftest import (
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
    get_static_diagnostic_statistics,
    get_timeseries_diagnostic_statistics,
)
from cider.featurizer.inference import (
    identify_daytime,
    identify_weekend,
    swap_caller_and_recipient,
    identify_and_tag_conversations,
    identify_active_days,
    get_number_of_contacts_per_caller,
    get_call_duration_stats,
    get_percentage_of_nocturnal_interactions,
    get_percentage_of_initiated_conversations,
)


class TestFeaturizerDependencies:

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
        filtered_data = filter_to_datetime(
            df,
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
    def test_get_static_diagnostic_statistics(self, data):
        df = pd.DataFrame(data)
        stats = get_static_diagnostic_statistics(df)

        assert stats.total_transactions == len(df)
        assert stats.num_unique_callers == df["caller_id"].nunique()
        assert stats.num_days == df["timestamp"].dt.date.nunique()
        if "recipient_id" in df.columns:
            assert stats.num_unique_recipients == df["recipient_id"].nunique()
        else:
            assert stats.num_unique_recipients == 0

    @pytest.mark.parametrize(
        "data",
        [
            CDR_DATA,
            MOBILE_DATA_USAGE_DATA,
            MOBILE_MONEY_TRANSACTION_DATA,
            RECHARGE_DATA,
        ],
    )
    def test_get_timeseries_diagnostic_statistics(self, data):
        df = pd.DataFrame(data)

        static_data = get_static_diagnostic_statistics(df)

        unique_days = df["timestamp"].dt.date.nunique()
        timeseries_stats = get_timeseries_diagnostic_statistics(df)
        assert set(timeseries_stats.columns).issubset(
            {
                "day",
                "transaction_type",
                "total_transactions",
                "num_unique_callers",
                "num_unique_recipients",
            }
        )
        assert timeseries_stats.day.nunique() == unique_days
        assert (
            static_data.total_transactions
            == timeseries_stats["total_transactions"].sum()
        )


class TestFeaturizerInference:

    def test_identify_daytime(self, spark):
        spark_cdr_data = spark.createDataFrame(pd.DataFrame(CDR_DATA))
        cdr_spark_with_daytime = identify_daytime(
            spark_cdr_data, day_start=12, day_end=17
        )
        cdr_with_daytime = cdr_spark_with_daytime.toPandas()

        assert "is_daytime" in cdr_with_daytime.columns
        assert cdr_with_daytime.is_daytime.values.tolist() == [0, 1, 1, 1, 0, 0]

        pd_cdr_data = pd.DataFrame(CDR_DATA).drop(columns=["timestamp"])
        spark_cdr_data_no_timestamp = spark.createDataFrame(pd_cdr_data)
        with pytest.raises(
            ValueError, match="Dataframe must contain 'timestamp' column"
        ):
            identify_daytime(spark_cdr_data_no_timestamp)

    def test_identify_weekend(self, spark):
        spark_cdr_data = spark.createDataFrame(pd.DataFrame(CDR_DATA))
        cdr_spark_with_weekend = identify_weekend(spark_cdr_data, weekend_days=[2, 6])
        cdr_with_weekend = cdr_spark_with_weekend.toPandas()

        assert "is_weekend" in cdr_with_weekend.columns
        assert cdr_with_weekend.is_weekend.values.tolist() == [0, 1, 1, 0, 0, 1]

        pd_cdr_data = pd.DataFrame(CDR_DATA).drop(columns=["timestamp"])
        spark_cdr_data_no_timestamp = spark.createDataFrame(pd_cdr_data)
        with pytest.raises(
            ValueError, match="Dataframe must contain 'timestamp' column"
        ):
            identify_weekend(spark_cdr_data_no_timestamp)

    def test_swap_caller_and_recipient(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)
        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_swapped = swap_caller_and_recipient(spark_cdr_data)
        pd_cdr_swapped = spark_cdr_swapped.toPandas()

        assert len(pd_cdr_swapped) == 2 * len(pd_cdr_data)
        assert set(pd_cdr_swapped.caller_id.unique()) == set(
            pd_cdr_swapped.recipient_id.unique()
        )
        assert "direction_of_transaction" in pd_cdr_swapped.columns
        assert set(pd_cdr_swapped.direction_of_transaction.unique()) == {
            "outgoing",
            "incoming",
        }

        for col in [
            "caller_id",
            "recipient_id",
            "caller_antenna_id",
            "recipient_antenna_id",
        ]:
            spark_cdr_no_col = spark.createDataFrame(pd_cdr_data.drop(columns=[col]))
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'recipient_id', 'caller_antenna_id', and 'recipient_antenna_id' columns",
            ):
                swap_caller_and_recipient(spark_cdr_no_col)

    def test_identify_and_tag_conversations(self, spark):
        conversations = {
            "caller_id": ["user_1"] * 6,
            "recipient_id": ["user_2"] * 6,
            "timestamp": pd.to_datetime(
                [
                    "2023-01-10 10:00:00",
                    "2023-01-10 10:30:00",
                    "2023-01-10 10:45:00",
                    "2023-01-11 13:10:00",
                    "2023-01-11 13:30:00",
                    "2023-01-11 13:55:00",
                ]
            ),
            "transaction_scope": ["domestic"] * 6,
            "transaction_type": ["text", "text", "call", "text", "text", "text"],
        }
        pd_cdr_data = pd.concat(
            [pd.DataFrame(CDR_DATA), pd.DataFrame(conversations)], ignore_index=True
        )
        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_tagged = identify_and_tag_conversations(spark_cdr_data, max_wait=3600)
        pd_cdr_tagged = spark_cdr_tagged.toPandas()

        assert "conversation" in pd_cdr_tagged.columns
        convo_times = pd_cdr_tagged["conversation"].dropna().unique()
        assert len(convo_times) == 5

        for col in ["caller_id", "recipient_id", "timestamp", "transaction_type"]:
            spark_cdr_no_col = spark.createDataFrame(pd_cdr_data.drop(columns=[col]))
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'recipient_id', 'timestamp', and 'transaction_type' columns",
            ):
                identify_and_tag_conversations(spark_cdr_no_col)

    def test_identify_active_days(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)
        pd_cdr_data.loc[:, "day"] = pd_cdr_data["timestamp"].dt.date

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)

        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_cdr_with_conversations = identify_and_tag_conversations(
            spark_cdr_with_weekend
        )
        spark_cdr_active_days = identify_active_days(spark_cdr_with_conversations)

        pd_cdr_active_days = spark_cdr_active_days.toPandas()

        assert set(
            [
                "active_days_all",
                "active_days_weekday",
                "active_days_weekend",
                "active_days_day",
                "active_days_night",
                "active_days_weekday_day",
                "active_days_weekday_night",
                "active_days_weekend_day",
                "active_days_weekend_night",
            ]
        ).issubset(set(pd_cdr_active_days.columns))
        assert pd_cdr_active_days.shape == (3, 10)
        assert pd_cdr_active_days.active_days_all.values.tolist() == [2, 2, 2]
        assert pd_cdr_active_days.active_days_weekday.values.tolist() == [2, 1, 2]
        assert pd_cdr_active_days.active_days_weekend.values.tolist() == [0, 1, 0]
        assert pd_cdr_active_days.active_days_day.values.tolist() == [2, 2, 1]
        assert pd_cdr_active_days.active_days_night.values.tolist() == [0, 0, 1]
        assert pd_cdr_active_days.active_days_weekday_day.values.tolist() == [2, 1, 1]
        assert pd_cdr_active_days.active_days_weekday_night.values.tolist() == [0, 0, 1]
        assert pd_cdr_active_days.active_days_weekend_day.values.tolist() == [0, 1, 0]
        assert pd_cdr_active_days.active_days_weekend_night.values.tolist() == [0, 0, 0]

        pd_cdr_with_conversations = spark_cdr_with_conversations.toPandas()
        for col in ["caller_id", "timestamp", "is_daytime", "is_weekend", "day"]:
            spark_cdr_no_col = spark.createDataFrame(
                pd_cdr_with_conversations.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'timestamp', 'day', 'is_weekend', and 'is_daytime' columns",
            ):
                identify_active_days(spark_cdr_no_col)

    def test_get_number_of_contacts_per_caller(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)

        spark_cdr_num_contacts = get_number_of_contacts_per_caller(
            spark_cdr_with_weekend
        )

        pd_cdr_num_contacts = spark_cdr_num_contacts.toPandas()

        assert pd_cdr_num_contacts.shape == (3, 9)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_text_num_unique_contacts",
                "weekday_daytime_text_num_unique_contacts",
                "weekday_nighttime_call_num_unique_contacts",
                "weekday_daytime_call_num_unique_contacts",
                "weekend_nighttime_text_num_unique_contacts",
                "weekend_daytime_text_num_unique_contacts",
                "weekend_nighttime_call_num_unique_contacts",
                "weekend_daytime_call_num_unique_contacts",
            ]
        ) == set(pd_cdr_num_contacts.columns)

        pd_cdr_with_weekend = spark_cdr_with_weekend.toPandas()
        for col in [
            "caller_id",
            "recipient_id",
            "is_weekend",
            "is_daytime",
            "transaction_type",
        ]:
            spark_cdr_data_missing = spark.createDataFrame(
                pd_cdr_with_weekend.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'recipient_id', 'is_weekend', 'is_daytime', and 'transaction_type' columns",
            ):
                get_number_of_contacts_per_caller(spark_cdr_data_missing)

    def test_get_call_duration_stats(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)

        spark_cdr_call_stats = get_call_duration_stats(spark_cdr_with_weekend)

        pd_cdr_call_stats = spark_cdr_call_stats.toPandas()

        assert pd_cdr_call_stats.shape == (3, 29)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_avg_call_duration",
                "weekend_nighttime_avg_call_duration",
                "weekday_daytime_avg_call_duration",
                "weekend_daytime_avg_call_duration",
                "weekday_nighttime_median_call_duration",
                "weekend_nighttime_median_call_duration",
                "weekday_daytime_median_call_duration",
                "weekend_daytime_median_call_duration",
                "weekday_nighttime_max_call_duration",
                "weekend_nighttime_max_call_duration",
                "weekday_daytime_max_call_duration",
                "weekend_daytime_max_call_duration",
                "weekday_nighttime_min_call_duration",
                "weekend_nighttime_min_call_duration",
                "weekday_daytime_min_call_duration",
                "weekend_daytime_min_call_duration",
                "weekday_nighttime_stddev_call_duration",
                "weekend_nighttime_stddev_call_duration",
                "weekday_daytime_stddev_call_duration",
                "weekend_daytime_stddev_call_duration",
                "weekday_nighttime_skewness_call_duration",
                "weekend_nighttime_skewness_call_duration",
                "weekday_daytime_skewness_call_duration",
                "weekend_daytime_skewness_call_duration",
                "weekday_nighttime_kurtosis_call_duration",
                "weekend_nighttime_kurtosis_call_duration",
                "weekday_daytime_kurtosis_call_duration",
                "weekend_daytime_kurtosis_call_duration",
            ]
        ) == set(pd_cdr_call_stats.columns)

        pd_cdr_with_weekend = spark_cdr_with_weekend.toPandas()
        for col in [
            "caller_id",
            "is_weekend",
            "is_daytime",
            "transaction_type",
            "duration",
        ]:
            spark_cdr_data_missing = spark.createDataFrame(
                pd_cdr_with_weekend.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'transaction_type', 'is_weekend', 'is_daytime', and 'duration' columns",
            ):
                get_call_duration_stats(spark_cdr_data_missing)

    def test_get_percentage_nocturnal_interactions(self, spark):
        cdr_data = {
            "caller_id": ["caller_1"] * 3 + ["caller_2"] * 3,
            "recipient_id": ["recipient_1"] * 6,
            "caller_antenna_id": ["antenna_1", "antenna_2"] * 3,
            "recipient_antenna_id": ["antenna_3", "antenna_4"] * 3,
            "timestamp": pd.to_datetime(
                [
                    "2023-01-01 10:00:00",
                    "2023-01-02 12:00:00",
                    "2023-01-02 14:00:00",
                    "2023-01-04 22:00:00",
                    "2023-01-05 18:00:00",
                    "2023-01-06 21:00:00",
                ]
            ),
            "duration": [300, 200, 400, 100, 250, 150],
            "transaction_type": ["text", "call"] * 3,
            "transaction_scope": ["domestic"] * 2
            + ["international"] * 2
            + ["other"] * 2,
        }
        pd_cdr_data = pd.DataFrame(cdr_data)

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)

        spark_cdr_nocturnal_calls = get_percentage_of_nocturnal_interactions(
            spark_cdr_with_weekend
        )

        pd_cdr_nocturnal_calls = spark_cdr_nocturnal_calls.toPandas()

        assert pd_cdr_nocturnal_calls.shape == (2, 5)
        assert pd_cdr_nocturnal_calls.filter(like="nocturnal").sum(
            1
        ).tolist() == pytest.approx([66.67, 0.0], rel=1e-2)
        assert set(
            [
                "caller_id",
                "weekday_text_percentage_nocturnal_interactions",
                "weekend_text_percentage_nocturnal_interactions",
                "weekday_call_percentage_nocturnal_interactions",
                "weekend_call_percentage_nocturnal_interactions",
            ]
        ) == set(pd_cdr_nocturnal_calls.columns)

        pd_cdr_with_weekend = spark_cdr_with_weekend.toPandas()
        for col in ["caller_id", "is_daytime", "is_weekend", "transaction_type"]:
            spark_cdr_data_missing = spark.createDataFrame(
                pd_cdr_with_weekend.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'is_daytime', 'is_weekend' and 'transaction_type' columns",
            ):
                get_percentage_of_nocturnal_interactions(spark_cdr_data_missing)

    def test_get_percentage_of_initiated_conversations(self, spark):
        conversations = {
            "caller_id": ["user_1"] * 6,
            "recipient_id": ["user_2"] * 6,
            "timestamp": pd.to_datetime(
                [
                    "2023-01-10 10:00:00",
                    "2023-01-10 10:30:00",
                    "2023-01-10 10:45:00",
                    "2023-01-11 13:10:00",
                    "2023-01-11 13:30:00",
                    "2023-01-11 13:55:00",
                ]
            ),
            "transaction_scope": ["domestic"] * 6,
            "transaction_type": ["text", "text", "call", "text", "text", "text"],
        }
        pd_cdr_data = pd.concat(
            [pd.DataFrame(CDR_DATA), pd.DataFrame(conversations)], ignore_index=True
        )
        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_cdr_swapped = swap_caller_and_recipient(spark_cdr_with_weekend)
        spark_cdr_tagged = identify_and_tag_conversations(
            spark_cdr_swapped, max_wait=3600
        )
        spark_cdr_percentage_initiated = get_percentage_of_initiated_conversations(
            spark_cdr_tagged
        )
        pd_cdr_percentage_initiated = spark_cdr_percentage_initiated.toPandas()

        assert pd_cdr_percentage_initiated.shape == (6, 5)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_percentage_initiated_conversations",
                "weekend_nighttime_percentage_initiated_conversations",
                "weekday_daytime_percentage_initiated_conversations",
                "weekend_daytime_percentage_initiated_conversations",
            ]
        ) == set(pd_cdr_percentage_initiated.columns)

        pd_cdr_tagged = spark_cdr_tagged.toPandas()
        for col in [
            "caller_id",
            "timestamp",
            "conversation",
            "is_weekend",
            "is_daytime",
            "direction_of_transaction",
        ]:
            spark_cdr_no_col = spark.createDataFrame(pd_cdr_tagged.drop(columns=[col]))
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'timestamp', 'conversation', 'is_weekend', 'is_daytime' and 'direction_of_transaction' columns",
            ):
                get_percentage_of_initiated_conversations(spark_cdr_no_col)
