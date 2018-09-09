from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data, get_open_close_data
import os
import logging
from Hyper_Setup import log_file_name_Setup
logger = logging.getLogger(log_file_name_Setup)

class Stock_Query:
    def __init__(self):

        self.period = "10Y"
        self.interval = 86400

        self.param = [
            # Dow Jones
            {
                'q': ".DJI",
                'x': "INDEXDJX",
            },
            # NYSE COMPOSITE (DJ)
            {
                'q': "NYA",
                'x': "INDEXNYSEGIS",
            },
            # NYSE COMPOSITE (DJ)
            {
                'q': "NYA",
                'x': "NYSE",
            },
            # S&P 500
            {
                'q': ".INX",
                'x': "INDEXSP",
            },
            {
                'q': ".INX",
                'x': "INDEXCBOE",
            },
            # S&P 500
            {
                'q': ".INX",
                'x': "NASD",
            }

        ]




    def query(self, stock_sym, update = False):
        for k in self.param:
            k['q'] = stock_sym


        df = get_prices_time_data(self.param, self.period, self.interval)
        # df = get_open_close_data(self.param, self.period)

        if not df.__len__():
            logger.info("Query for "+stock_sym+" returned Zero Rows")
        return df
