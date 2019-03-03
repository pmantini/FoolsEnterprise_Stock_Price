from Stock_List import Stock_List
from Stock import Stock
import logging
import datetime
import pandas as pd
from Hyper_Setup import db_folder, log_file_name_Setup
import os

from App_Top_Movers import app as topmoversapp

company_data = pd.read_csv("companylist.csv")

# create logger
logger = logging.getLogger(log_file_name_Setup)
logger.setLevel(logging.INFO)

# create console handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

# create file handler
fh = logging.FileHandler(str(datetime.datetime.now()) + "-" + log_file_name_Setup + ".txt")
fh.setFormatter(formatter)
logger.addHandler(fh)

company_blacklist = pd.read_csv("blacklist.csv")
blacklist_symbols = [sym for sym in company_blacklist.Symbol]


def app():
    stock_list = Stock_List()
    for k in blacklist_symbols:
        logger.info("%s is Blacklisted" % (k))

    for sym,name in zip(company_data.Symbol, company_data.Name):

        if sym in blacklist_symbols:
            logger.info("%s is Blacklisted - Skipping to next" % (name))
            pathname = os.path.join(db_folder+"/", sym+".db")
            logger.info(pathname)

            logger.info("Deleting db %s from %s" % (db_folder + "/" + sym + ".db", "list_stocks.db"))
            stock_list.delete_stock(sym)

            if os.path.exists(pathname):

                logger.info("Deleting db %s" % (db_folder+"/" + sym+".db"))
                os.remove(pathname)
            else:
                logger.info("DB %s does not exist" % (db_folder + "/" + sym + ".db"))

            continue

        logger.info("Adding %s to list of stocks to track" % (name))
        database_name = db_folder+"/"+sym+".db"
        stock_list.add_stock(sym, name, database_name)

    stock_list.data_commit()


    list_of_stocks_to_track = stock_list.list_of_stocks()


    for k in list_of_stocks_to_track:

        if k[0] in blacklist_symbols:
            logger.info("%s is Blacklisted - Skipping to next" % (k[0]))
            continue


        stock_name = k[0]
        logger.info("Querying Alphavantage for %s" % (stock_name))
        stock = Stock(stock_name)
        stock.update_alpha_vantage()
        stock.close()

    topmoversapp()


if __name__ == "__main__":
    app()