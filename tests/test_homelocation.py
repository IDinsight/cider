import pytest
import pandas as pd
from cider.homelocation.schemas import (
    CallDataRecordData,
    AntennaData,
    GeographicUnit,
    GetHomeLocationAlgorithm,
)
from shapely import Polygon
from cider.homelocation.inference import (
    _prepare_home_location_data,
    _infer_home_locations,
    get_home_locations,
    get_accuracy,
)
from cider.homelocation.dependencies import (
    _deduplicate_points_within_buffer,
    get_voronoi_tessellation,
)
import geopandas as gpd

CDR_DATA = {
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
ANTENNA_DATA = {
    "antenna_id": ["antenna_1", "antenna_2", "antenna_3"],
    "tower_id": ["antenna_1", "antenna_2", "antenna_3"],
    "latitude": [1.5001, 2.4987, 3.3467],
    "longitude": [1.8965, 2.4231, 3.0078],
}
SHAPEFILE_DATA = gpd.GeoDataFrame(
    {"region": ["region_1", "region_2", "region_3"]},
    geometry=[
        Polygon(
            [(1.1920, 1.1245), (4.4358, 1.2395), (4.3526, 4.9873), (1.1557, 4.7873)]
        ),
        Polygon(
            [(4.3467, 4.8236), (4.7957, 6.2368), (6.5823, 6.2366), (6.6757, 4.1905)]
        ),
        Polygon(
            [(0.5873, 3.6922), (0.2684, 0.1578), (3.6124, 3.3649), (3.9823, 0.2396)]
        ),
    ],
)
SHAPEFILE_DATA.set_crs("EPSG:4326", inplace=True)
SHAPEFILE_DATA["geometry"] = SHAPEFILE_DATA.buffer(0)

HOME_LOCATION_GT = pd.DataFrame(
    {
        "caller_id": ["caller_1", "caller_2", "caller_3"],
        "caller_antenna_id": ["antenna_1", "antenna_1", "antenna_2"],
        "region": ["region_1", "region_1", "region_2"],
    }
)

POINTS_DATA = gpd.GeoDataFrame(
    {
        "ids": ["a", "b", "c"],
        "geometry": gpd.points_from_xy(
            [0.0001, 0.1, 0.00003], [0.0004, 0.0004, 0.0004]
        ),
    }
)
POINTS_DATA = POINTS_DATA.set_crs(epsg=4326)
POINTS_DATA = POINTS_DATA.to_crs(epsg=3857)


def _get_cdr_data_payload(input: str = "base") -> pd.DataFrame:
    match input:
        case "base":
            return pd.DataFrame(CDR_DATA)
        case "invalid_field":
            cdr_data_invalid = CDR_DATA.copy()
            cdr_data_invalid["invalid_field"] = [1, 2, 3, 4, 5, 6]
            return pd.DataFrame(cdr_data_invalid)
        case "missing_field":
            cdr_data_missing = CDR_DATA.copy()
            cdr_data_missing.pop("duration")
            return pd.DataFrame(cdr_data_missing)
        case "invalid_transaction_type":
            cdr_data_invalid_type = CDR_DATA.copy()
            cdr_data_invalid_type["transaction_type"] = ["text", "invalid"] * 3
            return pd.DataFrame(cdr_data_invalid_type)
        case "invalid_transaction_scope":
            cdr_data_invalid_scope = CDR_DATA.copy()
            cdr_data_invalid_scope["transaction_scope"] = ["domestic", "invalid"] * 3
            return pd.DataFrame(cdr_data_invalid_scope)


def _get_antenna_data_payload(input: str = "base") -> pd.DataFrame:
    match input:
        case "base":
            return pd.DataFrame(ANTENNA_DATA)
        case "invalid_field":
            antenna_data_invalid = ANTENNA_DATA.copy()
            antenna_data_invalid["invalid_field"] = [1, 2, 3]
            return pd.DataFrame(antenna_data_invalid)
        case "missing_field":
            antenna_data_missing = ANTENNA_DATA.copy()
            antenna_data_missing.pop("latitude")
            return pd.DataFrame(antenna_data_missing)
        case "missing_tower_id":
            antenna_data_missing_tower = ANTENNA_DATA.copy()
            antenna_data_missing_tower.pop("tower_id")
            return pd.DataFrame(antenna_data_missing_tower)
        case "renamed_tower_id":
            antenna_data_renamed = ANTENNA_DATA.copy()
            antenna_data_renamed["tower_id"] = ["tower_1", "tower_2", "tower_3"]
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
            ("base", "base", GeographicUnit.SHAPEFILE),
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
            shapefile_data=(
                SHAPEFILE_DATA if geographic_unit == GeographicUnit.SHAPEFILE else None
            ),
        )
        if geographic_unit == GeographicUnit.SHAPEFILE:
            prepared_data.drop(columns=["region", "index_right"], inplace=True)
        assert not prepared_data.empty
        assert prepared_data.shape == (6, 10)
        assert set(prepared_data.columns).issubset(
            set(cdr_df.columns).union(set(antenna_df.columns)) - {geographic_unit}
        )

    @pytest.mark.parametrize(
        "create_cdr_data_schema,create_antenna_data_schema,geographic_unit",
        [
            ("base", "renamed_tower_id", GeographicUnit.TOWER_ID),
            ("base", "missing_tower_id", GeographicUnit.TOWER_ID),
            ("base", "base", "incorrect_geographic_unit"),
            ("base", "base", GeographicUnit.SHAPEFILE),
        ],
        indirect=["create_cdr_data_schema", "create_antenna_data_schema"],
    )
    def test_prepare_home_location_data_invalid_data(
        self, create_cdr_data_schema, create_antenna_data_schema, geographic_unit
    ):
        cdr_df = create_cdr_data_schema
        antenna_df = create_antenna_data_schema

        with pytest.raises(ValueError):
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

        # Infer home locations
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

        # Infer home locations with additional columns to keep
        inferred_locations = _infer_home_locations(
            prepared_data=prepared_data,
            algorithm=algorithm,
            spark_session=spark,
            additional_columns_to_keep=["transaction_type"],
        )
        assert not inferred_locations.empty
        assert inferred_locations.shape == (3, 4)
        assert set(inferred_locations.columns) == {
            "caller_id",
            "caller_antenna_id",
            expected_output,
            "transaction_type",
        }
        assert inferred_locations[expected_output].tolist() == [1, 1, 1]

        with pytest.raises(ValueError):
            inferred_locations = _infer_home_locations(
                prepared_data=prepared_data,
                algorithm=algorithm,
                spark_session=spark,
                additional_columns_to_keep=["some_invalid_column"],
            )

    @pytest.mark.parametrize(
        "create_cdr_data_schema,create_antenna_data_schema,column_to_merge_on,column_to_measure_on",
        [
            ("base", "base", "caller_id", "caller_antenna_id"),
            ("base", "base", "caller_id", "region"),
            ("base", "base", "caller_antenna_id", "region"),
        ],
        indirect=["create_cdr_data_schema", "create_antenna_data_schema"],
    )
    def test_home_location_inference_accuracy(
        self,
        spark,
        create_cdr_data_schema,
        create_antenna_data_schema,
        column_to_merge_on,
        column_to_measure_on,
    ):
        cdr_data = create_cdr_data_schema
        antenna_data = create_antenna_data_schema
        home_locations = get_home_locations(
            validated_cdr_data=cdr_data,
            validated_antenna_data=antenna_data,
            geographic_unit=GeographicUnit.SHAPEFILE,
            algorithm=GetHomeLocationAlgorithm.COUNT_DAYS,
            shapefile_data=SHAPEFILE_DATA,
            spark_session=spark,
            additional_columns_to_keep=["region"],
        )
        accuracy_table = get_accuracy(
            inferred_home_locations=home_locations,
            groundtruth_home_locations=HOME_LOCATION_GT,
            column_to_merge_on=column_to_merge_on,
            column_to_measure_on=column_to_measure_on,
        )
        assert not accuracy_table.empty
        assert set(accuracy_table.columns) == {
            column_to_measure_on + "_groundtruth",
            column_to_measure_on + "_inferred",
        } | {"recall", "precision", "is_correct"}
        assert accuracy_table.shape[0] >= 1

    @pytest.mark.parametrize(
        "create_cdr_data_schema,create_antenna_data_schema,column_to_merge_on,column_to_measure_on",
        [
            ("base", "base", "some_invalid_id", "caller_antenna_id"),
            ("base", "base", "caller_id", "some_invalid_id"),
        ],
        indirect=["create_cdr_data_schema", "create_antenna_data_schema"],
    )
    def test_home_location_inference_accuracy_failure(
        self,
        spark,
        create_cdr_data_schema,
        create_antenna_data_schema,
        column_to_merge_on,
        column_to_measure_on,
    ):
        cdr_data = create_cdr_data_schema
        antenna_data = create_antenna_data_schema
        home_locations = get_home_locations(
            validated_cdr_data=cdr_data,
            validated_antenna_data=antenna_data,
            geographic_unit=GeographicUnit.SHAPEFILE,
            algorithm=GetHomeLocationAlgorithm.COUNT_DAYS,
            shapefile_data=SHAPEFILE_DATA,
            spark_session=spark,
            additional_columns_to_keep=["region"],
        )
        with pytest.raises(ValueError):
            get_accuracy(
                inferred_home_locations=home_locations,
                groundtruth_home_locations=HOME_LOCATION_GT,
                column_to_merge_on=column_to_merge_on,
                column_to_measure_on=column_to_measure_on,
            )


class TestHomeLocationDependencies:

    def test_deduplication_code(self):
        deduplicated_points = _deduplicate_points_within_buffer(
            xy_points=POINTS_DATA, points_id_col="ids", buffer_distance=1e4
        )
        assert not deduplicated_points.empty
        assert len(deduplicated_points) == 2
        assert set(deduplicated_points["ids"]) == {"a", "b"}

    def test_voronoi_tesellation_code(self):
        antenna_data = pd.DataFrame(ANTENNA_DATA)
        antenna_data = gpd.GeoDataFrame(
            antenna_data,
            geometry=gpd.points_from_xy(
                antenna_data["longitude"], antenna_data["latitude"]
            ),
            crs="EPSG:4326",
        )
        voronoi_gdf = get_voronoi_tessellation(
            xy_points=antenna_data,
            boundary_shapefile=SHAPEFILE_DATA,
            points_id_col="antenna_id",
            buffer_distance_for_deduplication=1e-6,
        )
        assert not voronoi_gdf.empty
        assert len(voronoi_gdf) == 3
        assert set(voronoi_gdf["antenna_id"]) == {"antenna_1", "antenna_2", "antenna_3"}
