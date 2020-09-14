import os
import pandas as pd
from FE_Stock.FE_DB_Models.FE_Stock_List import FE_Stock_List
from FE_Stock.FE_DB_Models.FE_Stock import FE_Stock
from FE_Stock.FE_Quandl.FE_Quandl import FE_Quandl
from setup import database_folder, db_stock_list, table_name
from setup import quandl_api_key_file

class app_setup:
    def __init__(self, companylist, blacklist):
        self.list_of_company = self.load_csv(companylist)
        self.list_of_black = self.load_csv(blacklist)

        self.stock_list_db = os.path.join(database_folder, db_stock_list)
        self.fe_quandl = FE_Quandl(quandl_api_key_file)

    def load_csv(self, filename):
        data = pd.read_csv(filename)

        for ind, sym, name in zip(data.index, data.Symbol, data.Name):

            if "." in sym:
                print("Replacing %s symbol with %s" % (data.loc[ind].Symbol, data.loc[ind].Symbol.replace(".", "_")))
                data.loc[ind].Symbol = data.loc[ind].Symbol.replace(".", "_")

        return data

    def get_company_list(self):
        blacklist_symbols = [sym for sym in self.list_of_black.Symbol]
        for ind, sym, name in zip(self.list_of_company.index, self.list_of_company.Symbol, self.list_of_company.Name):
            if sym in blacklist_symbols:
                print("Blacklisted: Popping %s")
                self.list_of_company = self.list_of_company.drop(ind)

        return self.list_of_company

    def run(self):

        list_of_company = self.get_company_list()

        fe_stock_list = FE_Stock_List()
        print("Initiated %s" % self.stock_list_db)
        fe_stock_list.init(table_name)

        for sym, name in zip(list_of_company.Symbol, list_of_company.Name):
            print("Adding %s to list" % sym)
            try:
                fe_stock_list.add_stock(sym, name)
            except:
                print("%s Exists" % sym)

        list_of_stock = fe_stock_list.list_of_stocks()
        fe_stock_list.close()

        fe_quandl = FE_Quandl(filename=quandl_api_key_file)
        for k in list_of_stock:
            print(k[0], table_name)
            fe_stock = FE_Stock(k[0], table_name)
            fe_stock.init()
            try:
                data = fe_quandl.get(k[0])
            except:
                print("%s Not found; deleteing from list" % k[0])
                fe_stock_list = FE_Stock_List()
                fe_stock_list.init(table_name)
                fe_stock_list.delete_stock(k[0])
                fe_stock_list.close()
                continue

            for ind in data.index:
                stock_row = {"date": str(ind).split(" ")[0]}
                stock_row["open"] = data.loc[ind]['Open']
                stock_row["high"] = data.loc[ind]['High']
                stock_row["low"] = data.loc[ind]['Low']
                stock_row["close"] = data.loc[ind]['Close']
                stock_row["volume"] = data.loc[ind]['Volume']
                try:
                    fe_stock.add_stock_row(stock_row)
                except:
                    print("Date %s exists" % stock_row["date"])

            fe_stock.close()
            print("Added price data for %s to database" % k[0])



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("-l", "--list", dest="list",
                        help="specify the name of the file with list of stocks", metavar="LIST", default="companylist.csv")
    parser.add_argument("-b", "--black-list", dest="black_list",
                        help="specify the name of the file with list of blacklisted stocks", metavar="BLACKLIST", default="blacklist.csv")

    args = parser.parse_args()

    app = app_setup(args.list, args.black_list)
    app.run()