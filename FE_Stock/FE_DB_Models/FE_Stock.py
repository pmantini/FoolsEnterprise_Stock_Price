import sqlite3
import os
from setup import database_folder, db_stock_list
from .FE_Stock_List import FE_Stock_List

class FE_Stock:

    def __init__(self, sym, stock_list_table):

        self.fe_stock_list = FE_Stock_List()
        self.fe_stock_list.init(stock_list_table)
        self.stock_db = self.fe_stock_list.get_stocks(sym)[2]

        self.fe_stock_list.close()

        db_name = os.path.join(database_folder, self.stock_db)
        self.db = sqlite3.connect(db_name)
        self.table_name = "data"
        self.cursor = self.db.cursor()

    def get_db(self):
        return self.stock_db

    def init(self):
        self.cursor.execute(
            "create table if not exists %s (date VARCHAR(20), open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume FLOAT, CONSTRAINT stock_date_unique UNIQUE (date))" % (self.table_name))

    def add_stock_row(self, stock_row):
        try:
            self.cursor.execute("insert into %s (date, open, high, low, close, volume) values (\'%s\', \'%f\', \'%f\', \'%f\', \'%f\', \'%f\')"
                                % (self.table_name, stock_row["date"], stock_row["open"], stock_row["high"], stock_row["low"], stock_row["close"], stock_row["volume"]))
        except Exception as e:
            raise e

    def close(self):
        self.db.commit()
        self.db.close()

