from Stock_List import Stock_List
from Stock import Stock
from stock_api.batch_query_alpha_vantage import Stock_Query
import pandas as pd
from Hyper_Setup import db_folder, log_file_name_batch_update
import logging
import datetime
from App_Top_Movers import app as topmoverapp

# create logger
logger = logging.getLogger(log_file_name_batch_update)
logger.setLevel(logging.INFO)

# create console handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

# create file handler
fh = logging.FileHandler(str(datetime.datetime.now()) + "-" + log_file_name_batch_update + ".txt")
fh.setFormatter(formatter)
logger.addHandler(fh)


company_blacklist = pd.read_csv("blacklist.csv")

blacklist = [sym for sym in company_blacklist.Symbol]

def app():
    batch = 100

    stock_list = Stock_List()
    list_of_stocks = stock_list.list_of_stocks()

    i = 0

    for k in list_of_stocks:
        if k[0] in blacklist:
            logger.info("%s is Blacklisted - popping from list" % (k[0]))
            list_of_stocks.pop(i)
            i +=1


    list_of_batch_of_stocks = []
    for i in range(0, len(list_of_stocks), batch):
        list_of_batch_of_stocks += [list(list_of_stocks[i:min(i+batch, len(list_of_stocks))])]

    stock_batch = Stock_Query()
    for k in list_of_batch_of_stocks:
        stock_syms = [l[0] for l in k]
        query_res_data, meta_data  = stock_batch.query(stock_syms)
        # print(query_res_data)
        for k in query_res_data.index:
            item = {'symbol': query_res_data.loc[k]['1. symbol'],'date': query_res_data.loc[k]['4. timestamp'].split(' ')[0], 'price': query_res_data.loc[k]['2. price']}
            print(item)
            stock = Stock(item['symbol'])
            stock.append_to_database(item)
            stock.close()


    topmoverapp()

if __name__ == "__main__":
    app()