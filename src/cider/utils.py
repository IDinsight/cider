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


from pydantic import BaseModel, ValidationError
from pyspark.sql import DataFrame as SparkDataFrame
from pandas import DataFrame as PandasDataFrame


def validate_dataframe(
    df: SparkDataFrame | PandasDataFrame,
    required_schema: type[BaseModel],
    check_data_points: bool = False,
) -> None:
    """
    Validate that the dataframe has the required schema.

    Args:
        df: Spark or Pandas dataframe to validate
        required_schema: Pydantic BaseModel schema that the dataframe must conform to
        check_data_points: Whether to check each of the rows in the dataframe, in addition to the column names. Default is False.

    Raises:
        ValueError: If any of the required columns are missing from the dataframe
    """
    df_columns = set(df.columns)
    required_columns = set(
        [k for k, field in required_schema.model_fields.items() if field.is_required()]
    )
    missing_columns = required_columns - df_columns
    if missing_columns:
        raise ValueError(
            f"The following required columns are missing from the dataframe: {missing_columns}"
        )

    if check_data_points:
        if isinstance(df, SparkDataFrame):
            pandas_df = df.toPandas()
        else:
            pandas_df = df.copy()

        for index, row in pandas_df.iterrows():
            try:
                required_schema.model_validate(row.to_dict())
            except ValidationError as e:
                raise ValueError(
                    f"Row {index} does not conform to the required schema: {e}"
                ) from e
