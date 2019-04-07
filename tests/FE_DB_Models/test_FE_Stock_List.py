import unittest
import sqlite3

from setup import database_folder, db_stock_list

from FE_Stock.FE_DB_Models.FE_Stock_List import FE_Stock_List
import os

class TestFEStockList(unittest.TestCase):

    def setUp(self):
        self.fe_stock_list = FE_Stock_List()
        self.test_table_name = "test_table"

        self.db_name = os.path.join(database_folder, db_stock_list)
        self.db = sqlite3.connect(self.db_name)

        self.cursor = self.db.cursor()

    def test_init_creates_db_if_doesnot_exists(self):

        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        self.assertFalse(self.cursor.fetchone())

        self.fe_stock_list.init(self.test_table_name)
        self.fe_stock_list.close()

        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

        res = self.cursor.fetchall()
        self.assertTrue(res)

        table_exists = False
        for k in res:
            if self.test_table_name in k:
                table_exists = True

        self.assertTrue(table_exists)


    def test_init_creates_db_with_columns(self):
        rows_in_db = ["stock_symbol", "stock_name", "table_name"]

        self.fe_stock_list.init(self.test_table_name)
        self.fe_stock_list.close()

        self.cursor.execute("PRAGMA table_info('%s')" % self.test_table_name)

        res = self.cursor.fetchall()
        columns = [k[1] for k in res]
        for k in rows_in_db:
            self.assertTrue(k in columns)


    def test_add_row(self):
        stock  = ["sym", "name"]
        table_prefix_extention = ".db"
        stock.append(stock[0] + table_prefix_extention)

        self.fe_stock_list.init(self.test_table_name)
        self.fe_stock_list.add_stock(stock[0], stock[1])
        self.fe_stock_list.close()

        self.cursor.execute("SELECT * from %s Where stock_symbol=\'%s\'" % (self.test_table_name, stock[0]))

        result = self.cursor.fetchone()
        self.assertTrue(list(result) == stock)

    def test_add_row_fail_same_name(self):
        stock = ["sym", "name"]
        table_prefix_extention = ".db"
        stock.append(stock[0]+table_prefix_extention)

        self.fe_stock_list.init(self.test_table_name)
        self.fe_stock_list.add_stock(stock[0], stock[1])

        thrown = False
        try:
            self.fe_stock_list.add_stock(stock[0], stock[1])
        except:
            thrown = True

        self.assertTrue(thrown)
        self.fe_stock_list.close()

    def test_delete_row(self):
        stock = ["sym", "name"]
        table_prefix = "table_"
        stock.append(table_prefix + stock[0])

        self.cursor.execute(
            "create table if not exists %s (stock_symbol VARCHAR(10), stock_name TEXT, table_name TEXT, CONSTRAINT stock_name_unique UNIQUE (stock_symbol))" % (
                self.test_table_name))
        self.cursor.execute("insert into %s (stock_symbol, stock_name, table_name) values (\'%s\', \'%s\', \'%s\')" % (self.test_table_name, stock[0], stock[1], stock[2]))
        self.db.commit()

        self.fe_stock_list.init(self.test_table_name)
        self.fe_stock_list.delete_stock(stock[0])
        self.fe_stock_list.close()

        self.cursor.execute("SELECT * from %s Where stock_symbol=\'%s\'" % (self.test_table_name, stock[0]))

        result = self.cursor.fetchone()
        self.assertFalse(result)

    def test_delete_stock_passes_if_stock_does_not_exists(self):
        stock_sym = "does_not_exist"
        self.fe_stock_list.init(self.test_table_name)
        thrown = False
        try:
            self.fe_stock_list.delete_stock(stock_sym)
        except:
            thrown = True
        self.assertFalse(thrown)
        self.fe_stock_list.close()


    def test_get_stock_returns_correct_stock(self):

        stock = ["sym", "name"]
        table_prefix = "table_"
        stock.append(table_prefix + stock[0])

        self.cursor.execute(
            "create table if not exists %s (stock_symbol VARCHAR(10), stock_name TEXT, table_name TEXT, CONSTRAINT stock_name_unique UNIQUE (stock_symbol))" % (
                self.test_table_name))
        self.cursor.execute(
            "insert into %s (stock_symbol, stock_name, table_name) values (\'%s\', \'%s\', \'%s\')" % (
            self.test_table_name, stock[0], stock[1], stock[2]))
        self.db.commit()

        self.fe_stock_list.init(self.test_table_name)
        result = self.fe_stock_list.get_stocks(stock[0])

        self.assertTrue(list(result) == stock)
        self.fe_stock_list.close()

    def test_get_stock_returns_none_if_does_not_exist(self):

        stock = ["sym", "name"]
        table_prefix = "table_"
        stock.append(table_prefix + stock[0])

        self.cursor.execute(
            "create table if not exists %s (stock_symbol VARCHAR(10), stock_name TEXT, table_name TEXT, CONSTRAINT stock_name_unique UNIQUE (stock_symbol))" % (
                self.test_table_name))
        self.cursor.execute(
            "insert into %s (stock_symbol, stock_name, table_name) values (\'%s\', \'%s\', \'%s\')" % (
            self.test_table_name, stock[0], stock[1], stock[2]))
        self.db.commit()

        self.fe_stock_list.init(self.test_table_name)
        result = self.fe_stock_list.get_stocks("doesnotexis")

        self.assertTrue(result is None)
        self.fe_stock_list.close()


    def test_list_stock_returns_correct_row_count(self):

        stock1 = ["sym1", "name1", "table1"]
        stock2 = ["sym2", "name2", "table2"]

        self.cursor.execute(
            "create table if not exists %s (stock_symbol VARCHAR(10), stock_name TEXT, table_name TEXT, CONSTRAINT stock_name_unique UNIQUE (stock_symbol))" % (
                self.test_table_name))
        self.cursor.execute(
            "insert into %s (stock_symbol, stock_name, table_name) values (\'%s\', \'%s\', \'%s\')" % (
            self.test_table_name, stock1[0], stock1[1], stock1[2]))
        self.cursor.execute(
            "insert into %s (stock_symbol, stock_name, table_name) values (\'%s\', \'%s\', \'%s\')" % (
                self.test_table_name, stock2[0], stock2[1], stock2[2]))
        self.db.commit()

        self.fe_stock_list.init(self.test_table_name)
        result = self.fe_stock_list.list_of_stocks()

        self.assertTrue(len(result) == 2)
        self.fe_stock_list.close()


    def tearDown(self):
        self.cursor.execute("DROP TABLE IF EXISTS %s" % (db_stock_list))
        self.db.commit()
        self.db.close()
        os.remove(self.db_name)