import pytest
import pandas as pd
from cider.homelocation.schemas import (
    CallDataRecordData,
    AntennaData,
    GeographicUnit,
    GetHomeLocationAlgorithm,
)
from cider.homelocation.inference import (
    _prepare_home_location_data,
    _infer_home_locations,
)

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
    "transaction_scope": ["domestic"] * 2 + ["international"] * 2 + ["other"] * 2,
}
antenna_data = {
    "antenna_id": ["antenna_1", "antenna_2"],
    "tower_id": ["antenna_1", "antenna_2"],
    "latitude": [1.0, 2.0],
    "longitude": [3.0, 4.0],
}


def _get_cdr_data_payload(input: str = "base") -> pd.DataFrame:
    match input:
        case "base":
            return pd.DataFrame(cdr_data)
        case "invalid_field":
            cdr_data_invalid = cdr_data.copy()
            cdr_data_invalid["invalid_field"] = [1, 2, 3, 4, 5, 6]
            return pd.DataFrame(cdr_data_invalid)
        case "missing_field":
            cdr_data_missing = cdr_data.copy()
            cdr_data_missing.pop("duration")
            return pd.DataFrame(cdr_data_missing)
        case "invalid_transaction_type":
            cdr_data_invalid_type = cdr_data.copy()
            cdr_data_invalid_type["transaction_type"] = ["text", "invalid"] * 3
            return pd.DataFrame(cdr_data_invalid_type)
        case "invalid_transaction_scope":
            cdr_data_invalid_scope = cdr_data.copy()
            cdr_data_invalid_scope["transaction_scope"] = ["domestic", "invalid"] * 3
            return pd.DataFrame(cdr_data_invalid_scope)


def _get_antenna_data_payload(input: str = "base") -> pd.DataFrame:
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
        case "missing_tower_id":
            antenna_data_missing_tower = antenna_data.copy()
            antenna_data_missing_tower.pop("tower_id")
            return pd.DataFrame(antenna_data_missing_tower)
        case "renamed_tower_id":
            antenna_data_renamed = antenna_data.copy()
            antenna_data_renamed["tower_id"] = ["tower_1", "tower_2"]
            return pd.DataFrame(antenna_data_renamed)


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
        [
            "invalid_field",
            "missing_field",
            "invalid_transaction_type",
            "invalid_transaction_scope",
        ],
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
            AntennaData.model_validate(create_antenna_data_schema)

    @pytest.mark.parametrize(
        "create_cdr_data_schema,create_antenna_data_schema,geographic_unit",
        [
            ("base", "base", GeographicUnit.ANTENNA_ID),
            ("base", "base", GeographicUnit.TOWER_ID),
        ],
        indirect=["create_cdr_data_schema", "create_antenna_data_schema"],
    )
    def test_prepare_home_location_data(
        self, create_cdr_data_schema, create_antenna_data_schema, geographic_unit
    ):
        cdr_df = create_cdr_data_schema
        antenna_df = create_antenna_data_schema

        prepared_data = _prepare_home_location_data(
            validated_cdr_data=cdr_df,
            validated_antenna_data=antenna_df,
            geographic_unit=geographic_unit,
        )
        assert not prepared_data.empty
        assert prepared_data.shape == (6, 10)
        assert set(prepared_data.columns) == (
            set(cdr_df.columns).union(set(antenna_df.columns)) - {geographic_unit}
        )

    @pytest.mark.parametrize(
        "create_cdr_data_schema,create_antenna_data_schema,geographic_unit, error",
        [
            ("base", "renamed_tower_id", GeographicUnit.TOWER_ID, AssertionError),
            ("base", "missing_tower_id", GeographicUnit.TOWER_ID, AssertionError),
            ("base", "base", "incorrect_geographic_unit", ValueError),
        ],
        indirect=["create_cdr_data_schema", "create_antenna_data_schema"],
    )
    def test_prepare_home_location_data_invalid_data(
        self, create_cdr_data_schema, create_antenna_data_schema, geographic_unit, error
    ):
        cdr_df = create_cdr_data_schema
        antenna_df = create_antenna_data_schema

        with pytest.raises(error):
            _prepare_home_location_data(
                validated_cdr_data=cdr_df,
                validated_antenna_data=antenna_df,
                geographic_unit=geographic_unit,
            )

    @pytest.mark.parametrize(
        "create_cdr_data_schema,create_antenna_data_schema,algorithm,expected_output",
        [
            (
                "base",
                "base",
                GetHomeLocationAlgorithm.COUNT_TRANSACTIONS,
                "transaction_count",
            ),
            (
                "base",
                "base",
                GetHomeLocationAlgorithm.COUNT_DAYS,
                "transaction_days_count",
            ),
            (
                "base",
                "base",
                GetHomeLocationAlgorithm.COUNT_MODAL_DAYS,
                "transaction_modal_days_count",
            ),
        ],
        indirect=["create_cdr_data_schema", "create_antenna_data_schema"],
    )
    def test_inference_home_location(
        self,
        create_cdr_data_schema,
        create_antenna_data_schema,
        spark,
        algorithm,
        expected_output,
    ):
        cdr_df = create_cdr_data_schema
        antenna_df = create_antenna_data_schema

        prepared_data = _prepare_home_location_data(
            validated_cdr_data=cdr_df,
            validated_antenna_data=antenna_df,
            geographic_unit=GeographicUnit.ANTENNA_ID,
        )

        inferred_locations = _infer_home_locations(
            prepared_data=prepared_data,
            algorithm=algorithm,
            spark_session=spark,
        )
        assert not inferred_locations.empty
        assert inferred_locations.shape == (3, 3)
        assert set(inferred_locations.columns) == {
            "caller_id",
            "caller_antenna_id",
            expected_output,
        }
        assert inferred_locations[expected_output].tolist() == [1, 1, 1]
