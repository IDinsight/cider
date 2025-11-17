import pytest
from pyspark.sql import SparkSession
import geopandas as gpd
import pandas as pd

from shapely import Polygon


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master("local[1]").appName("pytest-spark").getOrCreate()
    )
    yield spark
    spark.stop()


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
