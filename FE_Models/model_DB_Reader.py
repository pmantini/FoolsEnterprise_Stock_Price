import sqlite3
import pandas as pd

from setup import log_file, database_folder, table_name
from FE_Stock.FE_DB_Models.FE_Stock_List import FE_Stock_List
from FE_Stock.FE_DB_Models.FE_Stock import FE_Stock

import logging
logging = logging.getLogger("main")

class DB_Ops:
    def __init__(self):

        feStocksList = FE_Stock_List()
        feStocksList.init(table_name)
        self.company_list = feStocksList.list_of_stocks()
        feStocksList.close()

        self.comp_list = [k[0] for k in self.company_list]

        self.db_folder = database_folder

        max_value = 0
        for k in self.comp_list:
            values_list,_ = self.get_values_company(k)
            if max_value < len(values_list):
                max_value = len(values_list)


        logging.info("%s total rows", max_value)

        self.max_rows = max_value


    def get_list_companies(self):
        return self.comp_list


    def get_values_company(self, company_sym, columns = "close"):
        feStock = FE_Stock(company_sym, table_name)
        all_items = feStock.fetch_all(columns)

        return [float(k[1]) for k in all_items], [k[0] for k in all_items]


    def get_max_rows(self):
        return self.max_rows


    def get_companies_count(self):
        return len(self.company_list)