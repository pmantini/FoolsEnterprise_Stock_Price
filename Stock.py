import sqlite3
from Stock_List import  Stock_List
# from stock_api.stock_query_alpha_vantage import Stock_Query
from stock_api.stock_query_google_finance import Stock_Query
from sqlite3 import IntegrityError

from Hyper_Setup import start_date

class Stock:
    def __init__(self, stock_sym):
        self.stock_sym = stock_sym

        stocks_row =  Stock_List()
        self.stocks_item = stocks_row.get_stocks(stock_sym)
        database_name = self.stocks_item[2]

        self.stocks = sqlite3.connect(database_name)
        self.cursor = self.stocks.cursor()
        self.table_name = "data"

        self.query_obj = Stock_Query()

        self.cursor.execute("create table if not exists %s (date TEXT, start_price TEXT, CONSTRAINT stock_date_unique UNIQUE (date))" % (self.table_name))


    def get_last_date(self):
        self.cursor.execute("SELECT * FROM %s ORDER BY date DESC LIMIT 1" % self.table_name)
        last_item = (self.cursor.fetchone())

        if not last_item:
            return []
        else:
            return last_item[0]

    def data_commit(self):
        self.stocks.commit()


    def update(self):
        data = []


        if not self.get_last_date():
            query_res_data = self.query_obj.query(self.stock_sym, update=False)
            query_res_data = query_res_data.dropna(axis='columns')

            for k_ts in query_res_data.index:
                k = k_ts.date().strftime('%Y-%m-%d')
                # print(k >= start_date)
                if k >= start_date:
                    # print(k, query_res_data.loc[k]['1. open'])
                    data += [{'date': k, 'price': query_res_data.loc[k][self.stock_sym+'_Open'][0]}]


        else:
            query_res_data = self.query_obj.query(self.stock_sym)

            for k_ts in query_res_data.index:
                k = k_ts.date().strftime('%m-%d-%Y')
                if k > self.get_last_date():

                    data += [{'date': k, 'price': query_res_data.loc[k][self.stock_sym + '_Open'][0]}]
        # print(data)
        for item in data:
            try:

                print("Adding %s, %s for %s" % (item['date'], item['price'], self.stock_sym))
                self.cursor.execute("INSERT INTO %s VALUES (\'%s\',\'%s\')" % (self.table_name, item['date'], item['price']))
            except IntegrityError:
                print("Date %s already added for %s!" % (item['date'], self.stock_sym))

        print("%s Updated!" % self.stock_sym)




    def close(self):
        self.data_commit()
        self.stocks.close()