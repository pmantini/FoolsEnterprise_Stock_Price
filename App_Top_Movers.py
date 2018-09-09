import datetime
import logging
from Hyper_Setup import db_folder, log_file_name_top_movers
import sqlite3

# create logger
logger = logging.getLogger(log_file_name_top_movers)
logger.setLevel(logging.INFO)

# create console handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

# create file handler
fh = logging.FileHandler(str(datetime.datetime.now()) + "-" + log_file_name_top_movers + ".txt")
fh.setFormatter(formatter)
logger.addHandler(fh)


from Stock_List import Stock_List
from Stock import Stock

stock_list = Stock_List()
change = []

for (sym, _, database) in stock_list.list_of_stocks():
    stock = Stock(sym)
    last_two_records =  stock.fetch_latest(2)

    if len(last_two_records) == 2:
        change += [(sym, (float(last_two_records[0][1]) - float(last_two_records[1][1])))]
    else:
        change += [(sym, '-')]
    stock.close()

sorted_change = sorted(change,key=lambda k: k[1])



db_name = "change.db"
changed_db = sqlite3.connect(db_folder+'/'+db_name)
cursor = changed_db.cursor()
table_name = "data"

cursor.execute("create table if not exists %s (stock_symbol TEXT, stock_change TEXT, CONSTRAINT stock_name_unique UNIQUE (stock_symbol))" % (table_name))

for k in sorted_change:
    logger.info("Adding %s, %s to change database" % (k[0], k[1]))
    cursor.execute("INSERT INTO %s VALUES (\'%s\',\'%s\')" % (table_name, k[0], k[1]))

stock_list.close()