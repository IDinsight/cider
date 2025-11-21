from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated


class DataDiagnosticStatistics(BaseModel):
    """
    Schema for data diagnostic statistics.
    """

    model_config = ConfigDict(from_attributes=True)

    total_transactions: Annotated[
        int, Field(description="Total number of transactions in the dataset")
    ]
    num_unique_callers: Annotated[
        int, Field(description="Number of unique caller IDs in the dataset")
    ]
    num_unique_recipients: Annotated[
        int, Field(description="Number of unique recipient IDs in the dataset")
    ]
    num_days: Annotated[
        int, Field(description="Number of unique days covered in the dataset")
    ]
