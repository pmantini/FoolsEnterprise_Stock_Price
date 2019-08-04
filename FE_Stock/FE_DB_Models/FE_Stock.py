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

    def delete_stock_row(self, date):
        print("Deleting: ", date)
        try:
            self.cursor.execute("delete from %s where date=\'%s\'" % (self.table_name, date))
        except Exception as e:
            raise e

    def fetch_all(self, column):
        if column == "all":
            self.cursor.execute("SELECT * FROM %s" % (self.table_name))
        else:
            self.cursor.execute("SELECT %s, %s FROM %s" % ("date", column, self.table_name))
        all_item = (self.cursor.fetchall())

        if not all_item:
            return []
        else:
            return all_item

    def fetch_by_date(self, date, column):

        if column == "all":
            self.cursor.execute("SELECT * FROM %s where date=\"%s\"" % (self.table_name, date))
        else:
            self.cursor.execute("SELECT %s, %s FROM %s where date=\"%s\"" % ("date", column, self.table_name, date))
        all_item = (self.cursor.fetchall())

        if not all_item:
            return []
        else:
            return all_item

    def fetch_latest(self, number_of_records=1):
        self.cursor.execute("SELECT * FROM %s ORDER BY date DESC LIMIT %d" % (self.table_name, number_of_records))
        all_item = (self.cursor.fetchall())

        if not all_item:
            return []
        else:
            return all_item

    def get_last_date(self):
        self.cursor.execute("SELECT * FROM %s ORDER BY date DESC LIMIT 1" % self.table_name)
        last_item = (self.cursor.fetchone())

        if not last_item:
            return []
        else:
            return last_item[0]

    def close(self):
        self.db.commit()
        self.db.close()

