from alpha_vantage.timeseries import TimeSeries
from random import choice
import os


class Stock_Query:
    def __init__(self):
        self.list_of_keys = ["XGWCYNOX99ZVR845", "FNWIERCF7YATSTZ2"]


    def get_key(self):
        return choice(self.list_of_keys)



    def query(self, stock_sym, update = True):
        ts = TimeSeries(key=self.get_key(), output_format='pandas')
        if update:
            query_result = ts.get_daily(symbol=stock_sym, outputsize="compact")
        else:
            query_result = ts.get_daily(symbol=stock_sym, outputsize="full", )
        return query_result

