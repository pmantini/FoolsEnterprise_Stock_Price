import quandl
from setup import start_date

class FE_Quandl:

    def __init__(self, filename):
        quandl.read_key(filename=filename)
        self.data_stream = "EOD/"

    def get_qunadl_key(self):
        return quandl.ApiConfig.api_key


    def get(self, stock_symbol):
        data = quandl.get(self.data_stream+stock_symbol, api_key=self.get_qunadl_key(), start_date=start_date)
        return data

    def filter(self, stock_symbol, start_date):
        data = quandl.get(self.data_stream+stock_symbol, api_key=self.get_qunadl_key(), start_date=start_date)
        return data
