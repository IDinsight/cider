from .schemas import GetHomeLocationAlgorithm, GeographicUnit
import pandas as pd
import geopandas as gpd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    count,
    desc_nulls_last,
    row_number,
    col,
    to_date,
    countDistinct,
)
from pyspark.sql.window import Window


# Prepare data for home location inference
def _prepare_home_location_data(
    validated_cdr_data: pd.DataFrame,
    validated_antenna_data: pd.DataFrame,
    geographic_unit: GeographicUnit,
    shapefile_data: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    """
    Prepare data for home location inference

    Args:
        validated_cdr_data: validated call data records
        validated_antenna_data: validated antenna data
        geographic_unit: geographic unit to use for home location inference
    Returns:
        prepared_data: prepared data for home location inference
    """
    match geographic_unit:
        case GeographicUnit.ANTENNA_ID:
            prepared_data = validated_cdr_data.merge(
                validated_antenna_data,
                left_on="caller_antenna_id",
                right_on="antenna_id",
                how="inner",
            )
            assert (
                not prepared_data.empty
            ), "Prepared data is empty after merging CDR and antenna data. Please check the input data."

            prepared_data.drop(columns=["antenna_id"], inplace=True)

        case GeographicUnit.TOWER_ID:
            assert (
                "tower_id" in validated_antenna_data.columns
            ), f"Antenna data must contain 'tower_id' column for geographic unit {GeographicUnit.TOWER_ID}."
            prepared_data = validated_cdr_data.merge(
                validated_antenna_data,
                left_on="caller_antenna_id",
                right_on="tower_id",
                how="inner",
            )
            assert (
                not prepared_data.empty
            ), "Prepared data is empty after merging CDR and antenna data. Please check the input data."
            prepared_data.drop(columns=["tower_id"], inplace=True)

        case GeographicUnit.SHAPEFILE:
            assert (
                shapefile_data is not None
            ), f"Shapefile data must be provided for geographic unit {GeographicUnit.SHAPEFILE}."

            antennas_gdf = gpd.GeoDataFrame(
                validated_antenna_data,
                geometry=gpd.points_from_xy(
                    validated_antenna_data.longitude, validated_antenna_data.latitude
                ),
                crs="EPSG:4326",
            )
            antennas = gpd.sjoin(antennas_gdf, shapefile_data, op="within", how="left")

            prepared_data = validated_cdr_data.merge(
                antennas.drop(columns=["geometry"]),
                left_on="caller_antenna_id",
                right_on="antenna_id",
                how="inner",
            )
            assert (
                not prepared_data.empty
            ), "Prepared data is empty after merging CDR and antenna data. Please check the input data."
            prepared_data.drop(columns=["antenna_id", "geometry"], inplace=True)

        case _:
            raise ValueError(f"Unsupported geographic unit: {geographic_unit}")

    return prepared_data


def _infer_home_locations(
    prepared_data: pd.DataFrame,
    algorithm: GetHomeLocationAlgorithm,
    spark_session: SparkSession,
) -> pd.DataFrame:
    """
    Infer home locations based on the specified algorithm

    Args:
        prepared_data: prepared data for home location inference
        algorithm: algorithm to use for home location inference
    Returns:
        home_locations: inferred home locations
    """
    prepared_data_spark = spark_session.createDataFrame(prepared_data)
    match algorithm:
        case GetHomeLocationAlgorithm.COUNT_TRANSACTIONS:
            grouped_data = prepared_data_spark.groupby(
                [
                    "caller_id",
                    "caller_antenna_id",
                ]
            ).agg(count("timestamp").alias("transaction_count"))

            window = Window.partitionBy("caller_id").orderBy(
                desc_nulls_last("transaction_count")
            )
            grouped_data = (
                grouped_data.withColumn("order", row_number().over(window))
                .where(col("order") == 1)
                .select(["caller_id", "caller_antenna_id", "transaction_count"])
            )

        case GetHomeLocationAlgorithm.COUNT_DAYS:
            prepared_data_spark = prepared_data_spark.withColumn(
                "day", to_date("timestamp")
            )
            grouped_data = prepared_data_spark.groupby(
                [
                    "caller_id",
                    "caller_antenna_id",
                ]
            ).agg(countDistinct("day").alias("transaction_days_count"))
            window = Window.partitionBy("caller_id").orderBy(
                desc_nulls_last("transaction_days_count")
            )
            grouped_data = (
                grouped_data.withColumn("order", row_number().over(window))
                .where(col("order") == 1)
                .select(["caller_id", "caller_antenna_id", "transaction_days_count"])
            )

        case GetHomeLocationAlgorithm.COUNT_MODAL_DAYS:
            prepared_data_spark = prepared_data_spark.withColumn(
                "day", to_date("timestamp")
            )
            grouped_data = prepared_data_spark.groupby(
                ["caller_id", "caller_antenna_id", "day"]
            ).agg(count("timestamp").alias("transactions_per_day_count"))
            window = Window.partitionBy(["caller_id", "day"]).orderBy(
                desc_nulls_last("transactions_per_day_count")
            )
            grouped_data = (
                grouped_data.withColumn("order", row_number().over(window))
                .where(col("order") == 1)
                .groupby(["caller_id", "caller_antenna_id"])
                .agg(count("order").alias("transaction_modal_days_count"))
            )
            window = Window.partitionBy("caller_id").orderBy(
                desc_nulls_last("transaction_modal_days_count")
            )
            grouped_data = (
                grouped_data.withColumn("order", row_number().over(window))
                .where(col("order") == 1)
                .select(
                    ["caller_id", "caller_antenna_id", "transaction_modal_days_count"]
                )
            )
        case _:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    grouped_data = grouped_data.toPandas()
    return grouped_data


def get_home_locations(
    spark_session: SparkSession,
    validated_cdr_data: pd.DataFrame,
    validated_antenna_data: pd.DataFrame,
    geographic_unit: GeographicUnit,
    algorithm: GetHomeLocationAlgorithm,
    shapefile_data: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    """
    Get home locations based on the specified parameters

    Args:
        spark_session: Spark session
        validated_cdr_data: validated CDR data
        validated_antenna_data: validated antenna data
        geographic_unit: geographic unit for home location inference
        algorithm: algorithm to use for home location inference
        shapefile_data: optional shapefile data for geographic boundaries

    Returns:
        DataFrame containing the inferred home locations
    """
    prepared_data = _prepare_home_location_data(
        validated_cdr_data, validated_antenna_data, geographic_unit, shapefile_data
    )
    home_locations = _infer_home_locations(prepared_data, algorithm, spark_session)
    return home_locations
