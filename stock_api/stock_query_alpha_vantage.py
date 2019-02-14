from alpha_vantage.timeseries import TimeSeries

from random import choice
from time import sleep
import os
import logging
from Hyper_Setup import log_file_name_Setup
logger = logging.getLogger(log_file_name_Setup)

class Stock_Query:
    def __init__(self):
        self.list_of_keys = ["XGWCYNOX99ZVR845", "FNWIERCF7YATSTZ2"]
        self.sleep_time = 60


    def get_key(self):
        return choice(self.list_of_keys)



    def query(self, stock_sym, update = True):
        try:
            ts = TimeSeries(key=self.get_key(), output_format='pandas')
            if update:
                query_result = ts.get_daily(symbol=stock_sym, outputsize="compact")
            else:
                query_result = ts.get_daily(symbol=stock_sym, outputsize="full", )

        except ValueError:
            logger.info("ValueError: Alpha Vantage: Reached Call Frequence Limit")
            logger.info("Sleeping for %d second!" % self.sleep_time)
            sleep(self.sleep_time)
            query_result = self.query(stock_sym, update)

        except KeyError:
            logger.info("KeyError: Alpha Vantage: Reached Call Frequence Limit")
            logger.info("Sleeping for %d second!" % self.sleep_time)
            sleep(self.sleep_time)
            query_result = self.query(stock_sym, update)

        return query_result

