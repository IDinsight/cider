from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated
from datetime import datetime
from enum import Enum


# Enums
class CallDataRecordTransactionType(str, Enum):
    TEXT = "text"
    CALL = "call"


class TransactionScope(str, Enum):
    DOMESTIC = "domestic"
    INTERNATIONAL = "international"
    OTHER = "other"


class GetHomeLocationAlgorithm(str, Enum):
    COUNT_TRANSACTIONS = "count_transactions"
    COUNT_DAYS = "count_days"
    COUNT_MODAL_DAYS = "count_modal_days"


class GeographicUnit(str, Enum):
    ANTENNA_ID = "antenna_id"
    TOWER_ID = "tower_id"
    SHAPEFILE = "shapefile"


# Data schemas
class CallDataRecordData(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    caller_id: Annotated[str, Field(description="Unique identifier for the caller")]
    recipient_id: Annotated[
        str, Field(description="Unique identifier for the recipient")
    ]
    timestamp: Annotated[datetime, Field(description="Timestamp of the call")]
    duration: Annotated[float, Field(description="Duration of the call in seconds")]
    transaction_type: Annotated[
        CallDataRecordTransactionType,
        Field(description="Type of transaction: text or call"),
    ]
    transaction_scope: Annotated[
        TransactionScope,
        Field(description="Scope of transaction: international, domestic or other"),
    ]
    caller_antenna_id: Annotated[
        str | None, Field(description="Identifier for the antenna handling the caller")
    ] = None
    recipient_antenna_id: Annotated[
        str | None,
        Field(description="Identifier for the antenna handling the recipient"),
    ] = None


class AntennaData(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    antenna_id: Annotated[str, Field(description="Unique identifier for the antenna")]
    tower_id: Annotated[
        str | None,
        Field(description="Identifier for the tower associated with the antenna"),
    ] = None
    latitude: Annotated[float, Field(description="Latitude of the antenna location")]
    longitude: Annotated[float, Field(description="Longitude of the antenna location")]
