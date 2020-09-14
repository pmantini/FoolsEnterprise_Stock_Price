import sqlite3

from setup import database_folder, db_stock_list


import os

class FE_Stock_List:

    def __init__(self):

        db_name = os.path.join(database_folder, db_stock_list)
        self.db = sqlite3.connect(db_name)
        self.table_name = None
        self.cursor = self.db.cursor()

    def init(self, table_name):
        self.table_name = table_name
        self.cursor.execute(
            "create table if not exists %s (stock_symbol VARCHAR(10), stock_name TEXT, table_name TEXT, CONSTRAINT stock_name_unique UNIQUE (stock_symbol))" % (
                self.table_name))


    def add_stock(self, stock_sym, stock_name):
        stock_table_exten = '.db'
        try:
            self.cursor.execute("insert into %s (stock_symbol, stock_name, table_name) values (\'%s\', \'%s\', \'%s\')" % (self.table_name, stock_sym, stock_name, stock_sym+stock_table_exten))
        except Exception as e:
            raise e

    def delete_stock(self, stock_sym):
        self.cursor.execute("delete from %s where stock_symbol=\'%s\'" % (self.table_name, stock_sym))

    def list_of_stocks(self):
        self.cursor.execute("SELECT * FROM %s" % (self.table_name))
        return self.cursor.fetchall()

    def get_stocks(self, stock_sym):
        self.cursor.execute("SELECT * FROM %s where stock_symbol=\'%s\'" % (self.table_name, stock_sym))
        return self.cursor.fetchone()

    def close(self):
        self.db.commit()
        self.db.close()

















