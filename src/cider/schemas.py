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


class MobileMoneyTransactionType(str, Enum):
    CASHIN = "cashin"
    CASHOUT = "cashout"
    P2P = "p2p"
    BILLPAY = "billpay"
    OTHER = "other"


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


class RechargeData(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    caller_id: Annotated[str, Field(description="Unique identifier for the caller")]
    timestamp: Annotated[datetime, Field(description="Timestamp of the recharge")]
    amount: Annotated[
        float, Field(description="Amount of the recharge in local currency")
    ]


class MobileDataUsageData(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    caller_id: Annotated[str, Field(description="Unique identifier for the caller")]
    timestamp: Annotated[datetime, Field(description="Timestamp of the data usage")]
    volume: Annotated[float, Field(description="Volume of data used in MB")]


class MobileMoneyTransactionData(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    caller_id: Annotated[str, Field(description="Unique identifier for the caller")]
    recipient_id: Annotated[
        str, Field(description="Unique identifier for the recipient")
    ]
    timestamp: Annotated[datetime, Field(description="Timestamp of the call")]
    transaction_type: Annotated[
        MobileMoneyTransactionType,
        Field(description="Type of transaction: text or call"),
    ]
    amount: Annotated[
        float, Field(description="Amount of the transaction in local currency")
    ]
    caller_balance_before: Annotated[
        float,
        Field(description="Caller's balance before the transaction in local currency"),
    ]
    caller_balance_after: Annotated[
        float,
        Field(description="Caller's balance after the transaction in local currency"),
    ]
    recipient_balance_before: Annotated[
        float,
        Field(
            description="Recipient's balance before the transaction in local currency"
        ),
    ]
    recipient_balance_after: Annotated[
        float,
        Field(
            description="Recipient's balance after the transaction in local currency"
        ),
    ]
