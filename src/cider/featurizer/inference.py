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

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import (
    col,
    count,
    countDistinct,
    hour,
    dayofweek,
    lit,
    lag,
    last,
    when,
    mean as pys_mean,
    expr,
    min as pys_min,
    max as pys_max,
    stddev_pop,
    skewness,
    kurtosis,
    sum as pys_sum,
)
from pyspark.sql.window import Window
from .schemas import DirectionOfTransactionEnum, AllowedPivotColumnsEnum
from .dependencies import _get_agg_columns


def identify_daytime(
    spark_df: SparkDataFrame, day_start: int = 7, day_end: int = 19
) -> SparkDataFrame:
    """
    Identify daytime records in the dataframe.

    Args:
        df: Dataframe with a 'timestamp' column
        day_start: Hour to start daytime (inclusive)
        day_end: Hour to end daytime (exclusive)

    Returns:
        df: Dataframe with additional 'is_daytime' column
    """
    if "timestamp" not in spark_df.columns:
        raise ValueError("Dataframe must contain 'timestamp' column")

    spark_df = spark_df.withColumn(
        "is_daytime",
        when(
            (hour(col("timestamp")) >= day_start) & (hour(col("timestamp")) < day_end),
            1,
        ).otherwise(0),
    )
    return spark_df


def identify_weekend(
    spark_df: SparkDataFrame,
    weekend_days: list[int] = [1, 7],
):
    """
    Identify weekend records in the dataframe.

    Args:
        spark_df: Dataframe with a 'timestamp' column
        weekend_days: List of integers representing weekend days (1=Sunday, 7=Saturday)
    Returns:
        df: Dataframe with additional 'is_weekend' column
    """
    if "timestamp" not in spark_df.columns:
        raise ValueError("Dataframe must contain 'timestamp' column")

    spark_df = spark_df.withColumn(
        "is_weekend",
        when((dayofweek(col("timestamp"))).isin(weekend_days), 1).otherwise(0),
    )
    return spark_df


def swap_caller_and_recipient(
    spark_df: SparkDataFrame,
) -> SparkDataFrame:
    """
    Swap caller and recipient columns in the dataframe and append the swapped rows.

    Args:
        spark_df: Dataframe with 'caller_id' and 'recipient_id' columns

    Returns:
        df: Dataframe with swapped caller and recipient columns
    """

    if not set(
        ["caller_id", "recipient_id", "recipient_antenna_id", "caller_antenna_id"]
    ).issubset(set(spark_df.columns)):
        raise ValueError(
            "Dataframe must contain 'caller_id', 'recipient_id', 'caller_antenna_id', and 'recipient_antenna_id' columns"
        )

    # Add a direction_of_transaction column to indicate incoming/outgoing
    spark_df = spark_df.withColumn(
        "direction_of_transaction", lit(DirectionOfTransactionEnum.OUTGOING.value)
    )

    # Create a copy with swapped caller and recipient columns and
    # direction_of_transaction set to incoming
    spark_df_copy = spark_df.select(
        col("recipient_id").alias("caller_id"),
        col("caller_id").alias("recipient_id"),
        col("caller_antenna_id").alias("recipient_antenna_id"),
        col("recipient_antenna_id").alias("caller_antenna_id"),
        *[
            col(c)
            for c in spark_df.columns
            if c
            not in [
                "caller_id",
                "recipient_id",
                "caller_antenna_id",
                "recipient_antenna_id",
            ]
        ],
    )
    spark_df_copy = spark_df_copy.withColumn(
        "direction_of_transaction", lit(DirectionOfTransactionEnum.INCOMING.value)
    )

    # Append the swapped dataframe to the original dataframe
    spark_df = spark_df.unionByName(spark_df_copy)

    return spark_df


def identify_and_tag_conversations(
    spark_df: SparkDataFrame, max_wait: int = 3600
) -> SparkDataFrame:
    """
    Add conversation ids to interactions in the dataframe.

    From bandicoot's documentation:
    "We define conversations as a series of text messages between the user and one contact.
    A conversation starts with either of the parties sending a text to the other.
    A conversation will stop if no text was exchanged by the parties for an hour or if one of the parties call the other.
    The next conversation will start as soon as a new text is send by either of the parties."
    This functions tags interactions with the conversation id they are part of: the id is the start unix time of the
    conversation.

    Args:
        spark_df: spark dataframe
        max_wait: time (in seconds) after which a conversation ends if no texts or calls have been exchanged

    Returns:
        spark_df: tagged spark dataframe
    """
    if not set(["caller_id", "recipient_id", "timestamp", "transaction_type"]).issubset(
        set(spark_df.columns)
    ):
        raise ValueError(
            "Dataframe must contain 'caller_id', 'recipient_id', 'timestamp', and 'transaction_type' columns"
        )

    window = Window.partitionBy("caller_id", "recipient_id").orderBy("timestamp")

    spark_df = (
        spark_df.withColumn(
            # Cast timestamp to long for time calculations
            "timestamp",
            col("timestamp").cast("long"),
            # Add previous transaction type and timestamp columns
        )
        .withColumn("prev_transaction_type", lag(col("transaction_type")).over(window))
        .withColumn(
            "prev_timestamp",
            lag(col("timestamp")).over(window),
            # Calculate time lapse since previous interaction
        )
        .withColumn(
            "time_lapse",
            col("timestamp") - col("prev_timestamp"),
            # Identify start of new conversations
        )
        .withColumn(
            "conversation",
            when(
                (col("transaction_type") == "text")
                & (
                    (col("prev_transaction_type") == "call")
                    | (col("prev_transaction_type").isNull())
                    | (col("time_lapse") >= max_wait)
                ),
                col("timestamp"),
            ),
            # Identify ongoing conversations
        )
        .withColumn(
            "conversation_last", last("conversation", ignorenulls=True).over(window)
        )
        .withColumn(
            "conversation",
            when(col("conversation").isNotNull(), col("conversation")).otherwise(
                when(col("transaction_type") == "text", col("conversation_last"))
            ),
        )
        # Convert conversation back to timestamp
        .withColumn("conversation", col("conversation").cast("timestamp"))
        # Also convert timestamp back if needed
        .withColumn("timestamp", col("timestamp").cast("timestamp"))
        # Drop intermediate columns
        .drop(
            "prev_transaction_type", "prev_timestamp", "time_lapse", "conversation_last"
        )
    )
    return spark_df


def identify_active_days(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Identify active days for each caller in the dataframe, disaggregated by type and time of day.

    Args:
        spark_df: Dataframe with 'caller_id' and 'timestamp' columns

    Returns:
        df: Dataframe with additional 'active_days' column
    """
    if not set(["caller_id", "timestamp", "day", "is_weekend", "is_daytime"]).issubset(
        spark_df.columns
    ):
        raise ValueError(
            "Dataframe must contain 'caller_id', 'timestamp', 'day', 'is_weekend', and 'is_daytime' columns"
        )

    out = spark_df.groupby("caller_id").agg(
        # Overall
        countDistinct("day").alias("active_days_all"),
        # By weekday/weekend
        countDistinct(when(col("is_weekend") == 0, col("day"))).alias(
            "active_days_weekday"
        ),
        countDistinct(when(col("is_weekend") == 1, col("day"))).alias(
            "active_days_weekend"
        ),
        # By day/night
        countDistinct(when(col("is_daytime") == 1, col("day"))).alias(
            "active_days_day"
        ),
        countDistinct(when(col("is_daytime") == 0, col("day"))).alias(
            "active_days_night"
        ),
        # By both (4 combinations)
        countDistinct(
            when((col("is_weekend") == 0) & (col("is_daytime") == 1), col("day"))
        ).alias("active_days_weekday_day"),
        countDistinct(
            when((col("is_weekend") == 0) & (col("is_daytime") == 0), col("day"))
        ).alias("active_days_weekday_night"),
        countDistinct(
            when((col("is_weekend") == 1) & (col("is_daytime") == 1), col("day"))
        ).alias("active_days_weekend_day"),
        countDistinct(
            when((col("is_weekend") == 1) & (col("is_daytime") == 0), col("day"))
        ).alias("active_days_weekend_night"),
    )

    return out


def get_number_of_contacts_per_caller(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Identify number of unique contacts per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'recipient_id', 'is_weekend', 'is_daytime', and 'transaction_type' columns

    Returns:
        df: Dataframe with num unique callers for each combination of is_weekend, is_daytime, and transaction_type
    """
    if not set(
        ["caller_id", "recipient_id", "is_weekend", "is_daytime", "transaction_type"]
    ).issubset(spark_df.columns):
        raise ValueError(
            "Dataframe must contain 'caller_id', 'recipient_id', 'is_weekend', 'is_daytime', and 'transaction_type' columns"
        )

    # Count distinct contacts per caller, disaggregated by type and time of day
    spark_df_unique_contacts = spark_df.groupby(
        "caller_id", "is_weekend", "is_daytime", "transaction_type"
    ).agg(countDistinct("recipient_id").alias("num_unique_contacts"))
    aggs = _get_agg_columns(
        "num_unique_contacts",
        cols_to_use_for_pivot=[
            AllowedPivotColumnsEnum.IS_WEEKEND,
            AllowedPivotColumnsEnum.IS_DAYTIME,
            AllowedPivotColumnsEnum.TRANSACTION_TYPE,
        ],
        agg_func=pys_sum,
    )
    pivoted_df = spark_df_unique_contacts.groupby("caller_id").agg(*aggs)

    return pivoted_df


def get_call_duration_stats(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Get call duration statistics per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'is_weekend', 'is_daytime', and 'transaction_type' columns

    Returns:
        df: Dataframe with call duration statistics columns for each weekday/weekend and day/nighttime combination.
    """
    if not set(
        ["caller_id", "transaction_type", "is_weekend", "is_daytime", "duration"]
    ).issubset(spark_df.columns):
        raise ValueError(
            "Dataframe must contain 'caller_id', 'transaction_type', 'is_weekend', 'is_daytime', and 'duration' columns"
        )

    filtered_df = spark_df.filter(col("transaction_type") == "call")

    stats_df = filtered_df.groupby(
        "caller_id", "is_weekend", "is_daytime", "transaction_type"
    ).agg(
        pys_mean("duration").alias("avg_call_duration"),
        expr("percentile(duration, 0.5)").alias("median_call_duration"),
        pys_max("duration").alias("max_call_duration"),
        pys_min("duration").alias("min_call_duration"),
        stddev_pop("duration").alias("stddev_call_duration"),
        skewness("duration").alias("skewness_call_duration"),
        kurtosis("duration").alias("kurtosis_call_duration"),
    )

    all_stats_aggs = []
    for stats_col in [
        "avg_call_duration",
        "median_call_duration",
        "max_call_duration",
        "min_call_duration",
        "stddev_call_duration",
        "skewness_call_duration",
        "kurtosis_call_duration",
    ]:
        aggs = _get_agg_columns(
            stats_col,
            cols_to_use_for_pivot=[
                AllowedPivotColumnsEnum.IS_WEEKEND,
                AllowedPivotColumnsEnum.IS_DAYTIME,
            ],
            agg_func=pys_sum,
        )
        all_stats_aggs.extend(aggs)

    pivoted_df = stats_df.groupby("caller_id").agg(*all_stats_aggs)

    return pivoted_df


def get_percentage_of_nocturnal_interactions(
    spark_df: SparkDataFrame,
) -> SparkDataFrame:
    """
    Get percentage of nocturnal interactions per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'is_daytime', 'transaction_type' columns

    Returns:
        df: Dataframe with percentage of nocturnal interactions column
    """
    if not set(["caller_id", "is_daytime", "is_weekend", "transaction_type"]).issubset(
        spark_df.columns
    ):
        raise ValueError(
            "Dataframe must contain 'caller_id', 'is_daytime', 'is_weekend' and 'transaction_type' columns"
        )

    count_df = spark_df.groupby("caller_id").agg(
        count("*").alias("total_interactions"),
        pys_sum(when(col("is_daytime") == 0, 1).otherwise(0)).alias(
            "nocturnal_interactions"
        ),
    )
    count_df = count_df.withColumn(
        "percentage_nocturnal_interactions",
        (col("nocturnal_interactions") / col("total_interactions")) * 100,
    ).select("caller_id", "percentage_nocturnal_interactions")

    count_df = spark_df.join(count_df, on="caller_id", how="inner")
    aggs = _get_agg_columns(
        "percentage_nocturnal_interactions",
        cols_to_use_for_pivot=[
            AllowedPivotColumnsEnum.IS_WEEKEND,
            AllowedPivotColumnsEnum.TRANSACTION_TYPE,
        ],
        agg_func=pys_mean,
    )
    pivoted_df = count_df.groupby("caller_id").agg(*aggs)

    return pivoted_df


def get_percentage_of_initiated_conversations(
    spark_df: SparkDataFrame,
) -> SparkDataFrame:
    """
    Get percentage of initiated conversations per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'timestamp', 'conversation', 'is_weekend', 'direction_of_transaction' and 'is_daytime' columns

    Returns:
        df: Dataframe with percentage of initiated conversations column
    """
    if not set(
        [
            "caller_id",
            "timestamp",
            "conversation",
            "is_weekend",
            "is_daytime",
            "direction_of_transaction",
        ]
    ).issubset(spark_df.columns):
        raise ValueError(
            "Dataframe must contain 'caller_id', 'timestamp', 'conversation', 'is_weekend', 'is_daytime' and 'direction_of_transaction' columns"
        )

    # TODO: this calculation is copied from deprecated.helpers.features.precent_initiated_conversations
    # but it seems to calculate the average number of initiated conversations per daytime / weekend convo
    # rather than the percentage. Keeping as is, but needs to be verified.
    convo_df = (
        spark_df.where(col("conversation") == col("timestamp"))
        .withColumn(
            "initiated_conversation",
            when(col("direction_of_transaction") == "outgoing", 1).otherwise(0),
        )
        .groupby("caller_id", "is_weekend", "is_daytime")
        .agg(
            pys_mean("initiated_conversation").alias(
                "percentage_initiated_conversations"
            )
        )
    )
    aggs = _get_agg_columns(
        "percentage_initiated_conversations",
        cols_to_use_for_pivot=[
            AllowedPivotColumnsEnum.IS_WEEKEND,
            AllowedPivotColumnsEnum.IS_DAYTIME,
        ],
        agg_func=pys_mean,
    )

    return convo_df.groupby("caller_id").agg(*aggs)
