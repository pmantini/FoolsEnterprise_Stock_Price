import App_Utils
import unittest
import datetime
from freezegun import freeze_time

class TestFEStocks(unittest.TestCase):
    def test_update_required_returns_false_for_today(self):
        with freeze_time("2019-04-04"):
            last_day = datetime.datetime.today().strftime('%Y-%m-%d')
            self.assertFalse(App_Utils.is_update_required(last_day))


    def test_update_required_returns_true_for_next_day(self):
        with freeze_time("2019-04-04"):
            last_day = (datetime.datetime.today() - datetime.timedelta(1)).strftime('%Y-%m-%d')
            self.assertTrue(App_Utils.is_update_required(last_day))


    def test_update_required_returns_false_for_weekend_1(self):
        with freeze_time("2019-04-06"):
            last_day = (datetime.datetime.today() - datetime.timedelta(1)).strftime('%Y-%m-%d')
            self.assertFalse(App_Utils.is_update_required(last_day))

    def test_update_required_returns_false_for_weekend_2(self):
        with freeze_time("2019-04-07"):
            last_day = (datetime.datetime.today() - datetime.timedelta(2)).strftime('%Y-%m-%d')
            self.assertFalse(App_Utils.is_update_required(last_day))

    def test_update_required_returns_false_for_day_afterweekend(self):
        with freeze_time("2019-04-08"):
            last_day = (datetime.datetime.today() - datetime.timedelta(2)).strftime('%Y-%m-%d')
            self.assertTrue(App_Utils.is_update_required(last_day))

    def test_update_required_returns_false_for_more_than_1_day(self):
        with freeze_time("2019-04-09"):
            last_day = (datetime.datetime.today() - datetime.timedelta(5)).strftime('%Y-%m-%d')
            self.assertTrue(App_Utils.is_update_required(last_day))