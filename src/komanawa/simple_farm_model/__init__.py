"""
A package for simple farm models.
"""
from komanawa.simple_farm_model.version import __version__
from komanawa.simple_farm_model.base_simple_farm_model import BaseSimpleFarmModel
from komanawa.simple_farm_model.simple_dairy_model import SimpleDairyModel, DairyModelWithSCScarcity, default_peak_cow
from komanawa.simple_farm_model.stock_rate_conversion import calc_full_farm_stock_rate