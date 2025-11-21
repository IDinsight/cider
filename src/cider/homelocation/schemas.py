from enum import Enum


class GetHomeLocationAlgorithm(str, Enum):
    COUNT_TRANSACTIONS = "count_transactions"
    COUNT_DAYS = "count_days"
    COUNT_MODAL_DAYS = "count_modal_days"


class GeographicUnit(str, Enum):
    ANTENNA_ID = "antenna_id"
    TOWER_ID = "tower_id"
    SHAPEFILE = "shapefile"
