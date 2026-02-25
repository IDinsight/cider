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

from pydantic import BaseModel, Field, model_validator
from typing import Annotated
from datetime import datetime
from enum import Enum
import numpy as np


# Enums
class CallDataRecordTransactionType(str, Enum):
    """
    Transaction type for call data records.
    """

    TEXT = "text"
    CALL = "call"


class TransactionScope(str, Enum):
    """
    Scope of the transaction for call data records:
    international, domestic, etc.
    """

    DOMESTIC = "domestic"
    INTERNATIONAL = "international"
    OTHER = "other"


class MobileMoneyTransactionType(str, Enum):
    """
    Transaction type for mobile money transactions.
    """

    CASHIN = "cashin"
    CASHOUT = "cashout"
    P2P = "p2p"
    BILLPAY = "billpay"
    OTHER = "other"


# Data schemas
class CallDataRecordData(BaseModel):
    """
    Data schema for call data records.
    """

    class Config:
        orm_mode = True

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
    """
    Data schema for antenna information.
    """

    class Config:
        orm_mode = True

    antenna_id: Annotated[str, Field(description="Unique identifier for the antenna")]
    tower_id: Annotated[
        str | None,
        Field(description="Identifier for the tower associated with the antenna"),
    ] = None
    latitude: Annotated[float, Field(description="Latitude of the antenna location")]
    longitude: Annotated[float, Field(description="Longitude of the antenna location")]


class RechargeData(BaseModel):
    """
    Data schema for recharge information.
    """

    class Config:
        orm_mode = True

    caller_id: Annotated[str, Field(description="Unique identifier for the caller")]
    timestamp: Annotated[datetime, Field(description="Timestamp of the recharge")]
    amount: Annotated[
        float, Field(description="Amount of the recharge in local currency")
    ]


class MobileDataUsageData(BaseModel):
    """
    Data schema for mobile data usage information.
    """

    class Config:
        orm_mode = True

    caller_id: Annotated[str, Field(description="Unique identifier for the caller")]
    timestamp: Annotated[datetime, Field(description="Timestamp of the data usage")]
    volume: Annotated[float, Field(description="Volume of data used in MB")]


class MobileMoneyTransactionData(BaseModel):
    """
    Data schema for mobile money transactions.
    """

    class Config:
        orm_mode = True

    caller_id: Annotated[str, Field(description="Unique identifier for the caller")]
    recipient_id: Annotated[
        str | None, Field(description="Unique identifier for the recipient")
    ]
    timestamp: Annotated[datetime, Field(description="Timestamp of the call")]
    transaction_type: Annotated[
        MobileMoneyTransactionType,
        Field(description="Type of transaction: cashin, cashout, etc."),
    ]
    amount: Annotated[
        float, Field(description="Amount of the transaction in local currency")
    ]
    caller_balance_before: Annotated[
        float | None,
        Field(description="Caller's balance before the transaction in local currency"),
    ] = None
    caller_balance_after: Annotated[
        float | None,
        Field(description="Caller's balance after the transaction in local currency"),
    ] = None
    recipient_balance_before: Annotated[
        float | None,
        Field(
            description="Recipient's balance before the transaction in local currency"
        ),
    ] = None
    recipient_balance_after: Annotated[
        float | None,
        Field(
            description="Recipient's balance after the transaction in local currency"
        ),
    ] = None

    @model_validator(mode="after")
    def check_balance_and_recipient_info(self) -> "MobileMoneyTransactionData":
        """
        Check that the balance information provided in the transaction is consistent with the transaction type and amount.
        For cashin and cashout transactions, recipient information should not be provided.
        For other transactions, if balance information is provided, it should be consistent with the transaction amount.
        """
        if (
            self.caller_balance_after is not None
            and self.caller_balance_before is not None
        ):
            if self.caller_balance_after != self.caller_balance_before - self.amount:
                raise ValueError(
                    f"Caller balance after transaction should be {self.caller_balance_before - self.amount}. Found {self.caller_balance_after} instead."
                )

        # Recipient should be None for cashin and cashout transactions
        if self.transaction_type in [
            MobileMoneyTransactionType.CASHIN,
            MobileMoneyTransactionType.CASHOUT,
        ]:
            if (
                self.recipient_id is not None
                or (
                    self.recipient_balance_after is not None
                    and ~np.isnan(self.recipient_balance_after)
                )
                or (
                    self.recipient_balance_before is not None
                    and ~np.isnan(self.recipient_balance_before)
                )
            ):
                raise ValueError(
                    f"Recipient ID and transaction balances should be None for transaction type {self.transaction_type}."
                )

        # For other transactions, recipient balances should match, if provided
        has_balance_before = self.recipient_balance_before is not None and not np.isnan(
            self.recipient_balance_before
        )
        has_balance_after = self.recipient_balance_after is not None and not np.isnan(
            self.recipient_balance_after
        )
        if has_balance_before != has_balance_after:
            raise ValueError(
                "If any recipient balance information is provided, all recipient balance fields must be provided."
            )
        if has_balance_before and has_balance_after:
            if self.recipient_id is None:
                raise ValueError(
                    "Recipient ID must be provided when recipient balance information is provided."
                )
            assert self.recipient_balance_before is not None
            assert self.recipient_balance_after is not None
            if (
                self.recipient_balance_after
                != self.recipient_balance_before + self.amount
            ):
                raise ValueError(
                    "Recipient balance after transaction does not match expected value."
                )

        return self
