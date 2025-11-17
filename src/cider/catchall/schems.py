from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated
from datetime import datetime
from enum import Enum


class MobileMoneyTransactionType(str, Enum):
    CASHIN = "cashin"
    CASHOUT = "cashout"
    P2P = "p2p"
    BILLPAY = "billpay"
    OTHER = "other"


class RechargesData(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    caller_id: Annotated[str, Field(description="Unique identifier for the caller")]
    timestamp: Annotated[datetime, Field(description="Timestamp of the recharge event")]
    amount: Annotated[float, Field(description="Amount recharged in local currency")]


class MobileData(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    caller_id: Annotated[str, Field(description="Unique identifier for the caller")]
    timestamp: Annotated[
        datetime, Field(description="Timestamp of the mobile data usage event")
    ]
    volume: Annotated[float, Field(description="Amount of mobile data used")]


class MobileMoneyData(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    caller_id: Annotated[str, Field(description="Unique identifier for the caller")]
    recipient_id: Annotated[
        str, Field(description="Unique identifier for the recipient")
    ]
    timestamp: Annotated[datetime, Field(description="Timestamp of the transaction")]
    amount: Annotated[
        float, Field(description="Amount of money transacted in local currency")
    ]
    transaction_type: Annotated[
        MobileMoneyTransactionType,
        Field(
            description="Type of transaction: cashin, cashout, p2p, billpay, or other"
        ),
    ]
    sender_balance_before: Annotated[
        float | None, Field(description="Sender balance before the transaction")
    ] = None
    sender_balance_after: Annotated[
        float | None, Field(description="Sender balance after the transaction")
    ] = None
    recipient_balance_before: Annotated[
        float | None, Field(description="Recipient balance before the transaction")
    ] = None
    recipient_balance_after: Annotated[
        float | None, Field(description="Recipient balance after the transaction")
    ] = None
