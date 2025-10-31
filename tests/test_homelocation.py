import pytest
import pandas as pd
from cider.homelocation.schemas import CallDataRecordData

cdr_data = {
    "caller_id": ["caller_1"] * 2 + ["caller_2"] * 2 + ["caller_3"] * 2,
    "recipient_id": ["recipient_1"] * 6,
    "caller_antenna_id": ["antenna_1", "antenna_2"] * 3,
    "timestamp": pd.to_datetime(
        [
            "2023-01-01 10:00:00",
            "2023-01-01 12:00:00",
            "2023-01-02 09:00:00",
            "2023-01-02 11:00:00",
            "2023-01-03 08:00:00",
            "2023-01-03 10:00:00",
        ]
    ),
    "duration": [300, 200, 400, 100, 250, 150],
    "transaction_type": ["text", "call"] * 3,
    "transaction_scope": ["domestic"] * 2 + ["international"] * 2 + ["domestic"] * 2,
}
antenna_data = {
    "antenna_id": ["antenna_1", "antenna_2"],
    "tower_id": ["tower_1", "tower_2"],
    "latitude": [1.0, 2.0],
    "longitude": [3.0, 4.0],
}


def _get_cdr_data_payload(input: str) -> pd.DataFrame:
    match input:
        case "base":
            return cdr_data
        case "invalid_field":
            cdr_data_invalid = cdr_data.copy()
            cdr_data_invalid["invalid_field"] = [1, 2, 3, 4, 5, 6]
            return pd.DataFrame(cdr_data_invalid)
        case "missing_field":
            cdr_data_missing = cdr_data.copy()
            cdr_data_missing.pop("duration")
            return pd.DataFrame(cdr_data_missing)


def _get_antenna_data_payload(input: str) -> pd.DataFrame:
    match input:
        case "base":
            return pd.DataFrame(antenna_data)
        case "invalid_field":
            antenna_data_invalid = antenna_data.copy()
            antenna_data_invalid["invalid_field"] = [1, 2]
            return pd.DataFrame(antenna_data_invalid)
        case "missing_field":
            antenna_data_missing = antenna_data.copy()
            antenna_data_missing.pop("latitude")
            return pd.DataFrame(antenna_data_missing)


class TestHomeLocationInference:
    @pytest.fixture
    def create_cdr_data_schema(self, request: pytest.FixtureRequest) -> dict:
        """Fixture to create experiment payload based on request parameter."""
        return _get_cdr_data_payload(request.param)

    @pytest.fixture
    def create_antenna_data_schema(self, request: pytest.FixtureRequest) -> dict:
        """Fixture to create experiment payload based on request parameter."""
        return _get_antenna_data_payload(request.param)

    @pytest.mark.parametrize(
        "create_cdr_data_schema",
        ["invalid_field", "missing_field"],
        indirect=["create_cdr_data_schema"],
    )
    def test_cdr_data_validation(self, create_cdr_data_schema):
        with pytest.raises(ValueError):
            CallDataRecordData.model_validate(create_cdr_data_schema)

    @pytest.mark.parametrize(
        "create_antenna_data_schema",
        ["invalid_field", "missing_field"],
        indirect=["create_antenna_data_schema"],
    )
    def test_antenna_data_validation(self, create_antenna_data_schema):
        with pytest.raises(ValueError):
            CallDataRecordData.model_validate(create_antenna_data_schema)

    #     if input_type == "base":
    #         # Should not raise any exception
    #         _ = _infer_home_locations(
    #             prepared_data=cdr_df,
    #             algorithm="count_transactions",
    #             spark=None
    #         )
    #     else:
    #         with pytest.raises(ValueError):
    #             _ = _infer_home_locations(
    #                 prepared_data=cdr_df,
    #                 algorithm="count_transactions",
    #                 spark=None
    #             )

    # def test_prepare_home_location_data(input_type):
    #     cdr_df = get_cdr_data_payload(input_type)
    #     antenna_df = pd.DataFrame(antenna_data)

    #     if input_type == "base":
    #         prepared_data = _prepare_home_location_data(
    #             validated_cdr_data=cdr_df,
    #             validated_antenna_data=antenna_df,
    #             geographic_unit="antenna_id"
    #         )
    #         assert not prepared_data.empty
    #         assert set(prepared_data.columns).issuperset(
    #             set(cdr_df.columns).union(set(antenna_df.columns)) - {"antenna_id"}
    #         )
    #     else:
    #         with pytest.raises(ValueError):
    #             _prepare_home_location_data(
    #                 validated_cdr_data=cdr_df,
    #                 validated_antenna_data=antenna_df,
    #                 geographic_unit="antenna_id"
    #             )
