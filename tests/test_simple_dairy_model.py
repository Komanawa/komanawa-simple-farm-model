"""
created matt_dumont 
on: 6/26/24
"""
import unittest
from komanawa.simple_farm_model.simple_dairy_model import SimpleDairyModel


class TestSimpleDairyModel(unittest.TestCase):

    def test_init_no_pg(self):
        all_months = [(7 - 1 + i) % 12 + 1 for i in range(12)] * 3
        SimpleDairyModel(all_months,
                         istate=[0, 0], pg=[0] * len(all_months), ifeed=[0, 0], imoney=[0, 0],
                         sup_feed_cost=1, product_price=10, monthly_input=True)

    def test_model_runs(self):
        all_months = [(7 - 1 + i) % 12 + 1 for i in range(12)] * 3
        model = SimpleDairyModel(all_months,
                         istate=[0, 0], pg=[0] * len(all_months), ifeed=[0, 0], imoney=[0, 0],
                         sup_feed_cost=1, product_price=10, monthly_input=True)
        model.run_model(printi=True)