import os
import pandas as pd
from FE_Stock.FE_DB_Models.FE_Stock_List import FE_Stock_List
from FE_Stock.FE_DB_Models.FE_Stock import FE_Stock
from FE_Stock.FE_Quandl.FE_Quandl import FE_Quandl
from setup import database_folder, db_stock_list, table_name, quandl_api_key_file
from App_Utils import is_update_required


class app_update:
    def __init__(self, companylist, blacklist, re_update = 0):
        self.list_of_company = self.load_csv(companylist)
        self.list_of_black = self.load_csv(blacklist)

        self.stock_list_db = os.path.join(database_folder, db_stock_list)
        self.fe_quandl = FE_Quandl(quandl_api_key_file)

        self.fe_stock_list = FE_Stock_List()
        self.fe_stock_list.init(table_name)

        self.re_update = re_update

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
                print("Blacklisted: Popping %s" % sym)

                self.list_of_company = self.list_of_company.drop(ind)

                result = self.fe_stock_list.get_stocks(sym)
                if result:
                    print("%s in database, deleteing" % sym)
                    self.fe_stock_list.delete_stock(sym)

                    try:
                        os.remove(os.path.join(database_folder, result[2]))
                        print("Deleting Stock Database %s" % os.path.join(database_folder, result[2]))
                    except:
                        print("%s does not exist" % os.path.join(database_folder, result[2]))
        self.fe_stock_list.close()

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

        fe_quandl = FE_Quandl(quandl_api_key_file)
        for stock in list_of_stock:
            print("Updating %s", stock[0])
            stock_sym = stock[0]
            fe_stock = FE_Stock(stock_sym, table_name)
            fe_stock.init()

            last_date = fe_stock.get_last_date()

            if is_update_required(last_date):
                if not len(last_date):
                    print(last_date)
                    print("No data availabel for %s", stock_sym)
                    data = fe_quandl.get(stock_sym)
                else:
                    data = fe_quandl.filter(stock_sym, last_date)

                for ind in data.index:
                    stock_row = {"date": str(ind).split(" ")[0]}
                    stock_row["open"] = data.loc[ind]['Open']
                    stock_row["high"] = data.loc[ind]['High']
                    stock_row["low"] = data.loc[ind]['Low']
                    stock_row["close"] = data.loc[ind]['Close']
                    stock_row["volume"] = data.loc[ind]['Volume']
                    try:
                        fe_stock.add_stock_row(stock_row)
                        print("Added %s for %s" % (stock_row["date"], stock_sym))
                    except:
                        print("Date %s exists for %s" % (stock_row["date"], stock_sym))

                fe_stock.close()
            else:
                print("%s is already Up to date, last date: %s" % (stock_sym, last_date))
                if self.c:
                    print("re_update flag set to True")
                    data = fe_quandl.filter(stock_sym, last_date)

                    for ind in data.index:
                        stock_row = {"date": str(ind).split(" ")[0]}
                        stock_row["open"] = data.loc[ind]['Open']
                        stock_row["high"] = data.loc[ind]['High']
                        stock_row["low"] = data.loc[ind]['Low']
                        stock_row["close"] = data.loc[ind]['Close']
                        stock_row["volume"] = data.loc[ind]['Volume']
                        try:
                            print("Deleting row %s" % stock_row["date"])
                            fe_stock.delete_stock_row(stock_row["date"])

                            fe_stock.add_stock_row(stock_row)
                            print("Added %s for %s" % (stock_row["date"], stock_sym))
                        except:
                            print("Date %s exists for %s" % (stock_row["date"], stock_sym))




if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("-l", "--list", dest="list",
                        help="specify the name of the file with list of stocks", metavar="LIST", default="companylist.csv")
    parser.add_argument("-b", "--black-list", dest="black_list",
                        help="specify the name of the file with list of blacklisted stocks", metavar="BLACKLIST", default="blacklist.csv")
    parser.add_argument("-ru", "--re-update", dest="re_update",
                        help="specify if the last row needs to be updated", metavar="REUPDATE",
                        default=0)

    args = parser.parse_args()

    app = app_update(args.list, args.black_list, int(args.re_update))
    app.run()