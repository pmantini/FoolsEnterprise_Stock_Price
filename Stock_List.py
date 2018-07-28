import sqlite3
from sqlite3 import IntegrityError
import logging
from Hyper_Setup import log_file_name
class Stock_List:
    def __init__(self):

        # create logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # create console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)

        # create file handler
        fh = logging.FileHandler(log_file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        self.logger = logger

        try:
            self.list_stocks = sqlite3.connect('Databases/list_stocks.db')
        except sqlite3.OperationalError:
            logger.error("error connecting to database")

        self.cursor = self.list_stocks.cursor()
        self.table_name = "list"
        self.cursor.execute("create table if not exists %s (stock_symbol TEXT, stock_name TEXT, database TEXT, CONSTRAINT stock_name_unique UNIQUE (stock_symbol))" % (self.table_name))

    def data_commit(self):
        self.list_stocks.commit()

    def add_stock(self, stock_sym, stock_name, database_name):
        try:
            self.cursor.execute("INSERT INTO %s VALUES (\'%s\',\'%s\', \'%s\')" % (self.table_name, stock_sym, stock_name, database_name))
        except IntegrityError:
            self.logger.info("Stock %s already exitst!" % stock_name)
            #print("Stock %s already exitst!" % stock_name)


    def delete_stock(self, stock_sym):
        self.cursor.execute("DELETE FROM %s WHERE stock_symbol=\'%s\'" % (self.table_name, stock_sym))


    def list_of_stocks(self):
        self.cursor.execute("SELECT * FROM %s" % (self.table_name))
        return self.cursor.fetchall()

    def get_stocks(self, stock_sym):
        self.cursor.execute("SELECT * FROM %s where stock_symbol=\'%s\'" % (self.table_name, stock_sym))
        return self.cursor.fetchone()

    def close(self):
        self.data_commit()
        self.list_stocks.close()

