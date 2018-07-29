from Stock_List import Stock_List
from Stock import Stock
import pandas as pd

company_data = pd.read_csv("companylist_test.csv")
database_folder = "Databases"
stock_list = Stock_List()


for sym,name in zip(company_data.Symbol, company_data.Name):
    print("Adding %s to list of stocks to track" % (name))
    database_name = database_folder+"/"+sym+".db"
    stock_list.add_stock(sym, name, database_name)

stock_list.data_commit()


list_of_stocks_to_track = stock_list.list_of_stocks()


for k in list_of_stocks_to_track:
    stock_name = k[0]
    stock = Stock(stock_name)
    stock.update()
    stock.close()