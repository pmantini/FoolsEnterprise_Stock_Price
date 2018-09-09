from Stock_List import Stock_List
from Stock import Stock
import logging
import datetime
from Hyper_Setup import log_file_name
import pandas as pd
from Hyper_Setup import db_folder
company_data = pd.read_csv("companylist.csv")

# create logger
logger = logging.getLogger("Stock_price")
logger.setLevel(logging.INFO)

# create console handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

# create file handler
fh = logging.FileHandler(str(datetime.datetime.now()) + "-" + log_file_name + ".txt")
fh.setFormatter(formatter)
logger.addHandler(fh)

stock_list = Stock_List()

for sym,name in zip(company_data.Symbol, company_data.Name):
    logger.info("Adding %s to list of stocks to track" % (name))    
    database_name = db_folder+"/"+sym+".db"
    stock_list.add_stock(sym, name, database_name)

stock_list.data_commit()


list_of_stocks_to_track = stock_list.list_of_stocks()


for k in list_of_stocks_to_track:
    stock_name = k[0]
    stock = Stock(stock_name)
    stock.update_alpha_vantage()
    stock.close()
