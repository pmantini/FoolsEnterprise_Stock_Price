from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data
import os


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
            # S&P 500
            {
                'q': ".INX",
                'x': "INDEXSP",
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

        return df
