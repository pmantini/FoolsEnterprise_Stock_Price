import unittest
import sqlite3
import os

from setup import database_folder, db_stock_list

from FE_Stock.FE_DB_Models.FE_Stock import FE_Stock, FE_Stock_List

class TestFEStocks(unittest.TestCase):
    def setUp(self):

        test_stock = ["sym", "name"]
        self.test_stock_list_table = "test_table"

        self.fe_stock_add_to_list = FE_Stock_List()
        self.fe_stock_add_to_list.init(self.test_stock_list_table)
        self.fe_stock_add_to_list.add_stock(test_stock[0], test_stock[1])
        self.fe_stock_add_to_list.close()

        self.fe_stock = FE_Stock(test_stock[0], self.test_stock_list_table)

        self.stock_db = "sym.db"
        self.db_name = os.path.join(database_folder, self.stock_db)
        self.db = sqlite3.connect(self.db_name)
        self.stock_table_name = "data"

        self.cursor = self.db.cursor()

    def test_stock_return_correct_table_name(self):
        self.assertTrue(self.stock_db == self.fe_stock.get_db())

    def test_stock_init_creates_db_if_doesnot_exists(self):

        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        self.assertFalse(self.cursor.fetchone())

        self.fe_stock.init()
        self.fe_stock.close()

        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

        res = self.cursor.fetchall()
        self.assertTrue(res)

        table_exists = False
        for k in res:
            if self.stock_table_name in k:
                table_exists = True

        self.assertTrue(table_exists)


    def test_stock_init_creates_db_with_columns(self):
        rows_in_db = ["date", "open", "high", "low", "close", "volume"]

        self.fe_stock.init()
        self.fe_stock.close()

        self.cursor.execute("PRAGMA table_info('%s')" % self.stock_table_name)

        res = self.cursor.fetchall()
        columns = [k[1] for k in res]

        for k in rows_in_db:
            self.assertTrue(k in columns)

    def test_stock_add_row(self):
        stock_row  = {"date":"2019-3-30", "open":30.5, "high":60.5, "low":-10.5, "close": 30.5, "volume":30000}

        self.fe_stock.init()
        self.fe_stock.add_stock_row(stock_row)
        self.fe_stock.close()

        self.cursor.execute("SELECT * from %s Where date=\'%s\'" % (self.stock_table_name, stock_row["date"]))

        result = self.cursor.fetchone()
        self.assertTrue(list(result) == list([stock_row["date"],stock_row["open"], stock_row["high"], stock_row["low"], stock_row["close"], stock_row["volume"]]))


    def test_stock_add_row_fail_same_name(self):
        stock_row  = {"date":"2019-3-30", "open":30.5, "high":60.5, "low":-10.5, "close": 30.5, "volume":30000}

        self.fe_stock.init()
        self.fe_stock.add_stock_row(stock_row)

        thrown = False
        try:
            self.fe_stock.add_stock_row(stock_row)
        except:
            thrown = True

        self.assertTrue(thrown)
        self.fe_stock.close()

    def test_stock_fetch_row_gets_correct_row_count(self):
        stock_row = {"date": "2019-3-30", "open": 30.5, "high": 60.5, "low": -10.5, "close": 30.5, "volume": 30000}

        self.fe_stock.init()
        self.fe_stock.add_stock_row(stock_row)


        result = self.fe_stock.fetch_latest(1)

        self.assertTrue(len(result) == 1)
        self.fe_stock.close()

    def test_stock_fetch_row_gets_correct_row_count_of_2(self):
        stock_row1 = {"date": "2019-3-30", "open": 30.5, "high": 60.5, "low": -10.5, "close": 30.5, "volume": 30000}
        stock_row2 = {"date": "2019-4-01", "open": 30.5, "high": 60.5, "low": -10.5, "close": 30.5, "volume": 30000}

        self.fe_stock.init()
        self.fe_stock.add_stock_row(stock_row1)
        self.fe_stock.add_stock_row(stock_row2)

        result = self.fe_stock.fetch_latest(2)

        self.assertTrue(len(result) == 2)
        self.fe_stock.close()

    def test_stock_fetch_last_day_gets_correct_date(self):
        stock_row1 = {"date": "2019-3-30", "open": 30.5, "high": 60.5, "low": -10.5, "close": 30.5, "volume": 30000}
        stock_row2 = {"date": "2019-4-01", "open": 30.5, "high": 60.5, "low": -10.5, "close": 30.5, "volume": 30000}

        self.fe_stock.init()
        self.fe_stock.add_stock_row(stock_row1)
        self.fe_stock.add_stock_row(stock_row2)

        result = self.fe_stock.get_last_date()

        self.assertTrue(result == stock_row2["date"])
        self.fe_stock.close()

    def test_delete_row_deletes_row(self):
        stock_row1 = {"date": "2019-3-30", "open": 30.5, "high": 60.5, "low": -10.5, "close": 30.5, "volume": 30000}
        stock_row2 = {"date": "2019-4-01", "open": 30.5, "high": 60.5, "low": -10.5, "close": 30.5, "volume": 30000}

        self.fe_stock.init()
        self.fe_stock.add_stock_row(stock_row1)
        self.fe_stock.add_stock_row(stock_row2)

        self.fe_stock.delete_stock_row(stock_row1["date"])
        self.fe_stock.close()

        self.cursor.execute("SELECT * from %s Where date=\'%s\'" % (self.stock_table_name, stock_row1["date"]))
        result = self.cursor.fetchone()
        self.assertTrue(result is None)


    def tearDown(self):
        # self.cursor.execute("DROP TABLE IF EXISTS %s" % (self.test_table_name))
        self.db.commit()
        self.db.close()
        os.remove(self.db_name)
        os.remove(os.path.join(database_folder, db_stock_list))



