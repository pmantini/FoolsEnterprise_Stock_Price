import sqlite3
from sqlite3 import IntegrityError

class Stock_List:
    def __init__(self, db_folder):
        self.database_folder = db_folder
        self.list_stocks = sqlite3.connect(db_folder+'/list_stocks.db')
        self.cursor = self.list_stocks.cursor()
        self.table_name = "list"
        self.cursor.execute("create table if not exists %s (stock_symbol TEXT, stock_name TEXT, database TEXT, CONSTRAINT stock_name_unique UNIQUE (stock_symbol))" % (self.table_name))

    def data_commit(self):
        self.list_stocks.commit()

    def add_stock(self, stock_sym, stock_name, database_name):
        try:
            self.cursor.execute("INSERT INTO %s VALUES (\'%s\',\'%s\', \'%s\')" % (self.table_name, stock_sym, stock_name, database_name))
        except IntegrityError:
            print("Stock %s already exitst!" % stock_name)


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

