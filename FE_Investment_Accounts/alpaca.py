import alpaca_trade_api as tradeapi
from setup import *

class FE_Alpaca:

    def __init__(self, live=False):


        if live:
            API_KEY = self.read_file(alpaca_api_key_file)
            API_SECRET = self.read_file(alpaca_api_secret_key_file)
            APCA_API_BASE_URL = "https://api.alpaca.markets"
        else:
            API_KEY = self.read_file(alpaca_api_paper_key_file)
            API_SECRET = self.read_file(alpaca_api_paper_secret_key_file)
            APCA_API_BASE_URL = "https://paper-api.alpaca.markets"

        self.alpaca = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, 'v2')
        self.account = self.alpaca.get_account()

    def get_account_details(self):
        return self.account.__dict__['_raw']


    def get_buying_power(self):
        return self.account.buying_power

    def get_positions(self, sym):
        return self.alpaca.get_position(sym)

    def get_all_positions(self):
        return self.alpaca.list_positions()

    def get_order(self):
        return self.alpaca.get_orders()

    def get_all_order(self):
        return self.alpaca.list_orders()

    def get_asset(self, sym):
        return self.alpaca.get_asset(sym)

    def is_asset_active(self, sym):
        return True if self.get_asset(sym).status == "active" else False

    def is_asset_tradable(self, sym):
        return self.get_asset(sym).tradable

    def list_assets(self):
        return self.alpaca.list_assets(status='active')

    def get_position(self,sym):
        return self.alpaca.get_position(sym)

    def create_list_assets(self, file = "alpaca_company.csv"):
        all_stocks = self.list_assets()
        list_stocks = []
        for k in all_stocks:
            list_stocks += [{"Symbol":k.symbol, "Name": k.name, "Sector":""}]
        import csv

        csv_columns = ['Symbol', 'Name', 'Sector']

        csv_file = file
        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in list_stocks:
                    writer.writerow(data)
        except IOError:
            print("I/O error")


    def order(self, sym, qty, side, limit, id, type="limit"):
        if not self.is_asset_active(sym):
            print("Asset %s not active" % sym)
            raise Exception("Asset %s not active" % sym)
        elif not self.is_asset_tradable(sym):
            print("Asset %s not tradable" % sym)
            raise Exception("Asset %s not tradable" % sym)
        else:
            return self.alpaca.submit_order(sym, qty, side, type, time_in_force="gtc", limit_price=limit, client_order_id=id)

    def cancel_order(self, id):
        self.alpaca.cancel_order(id)

    def liquidate_position(self, sym):
        self.alpaca.close_position(sym)

    def replace_order(self, order_id, qty=None, limit_price=None):
        self.alpaca.replace_order(order_id, qty=qty, limit_price=limit_price)


    def read_file(self, file):
        f = open(file, "r")
        if f.mode == 'r':
            return f.read()


