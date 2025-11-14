import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master("local[1]").appName("pytest-spark").getOrCreate()
    )
    yield spark
    spark.stop()
