import sqlite3
import pandas as pd

from setup import log_file, database_folder, table_name, start_date
from FE_Stock.FE_DB_Models.FE_Stock_List import FE_Stock_List
from FE_Stock.FE_DB_Models.FE_Stock import FE_Stock

import datetime
from datetime import timedelta

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
        logging.info("Total stocks: %s" % len(self.company_list) )
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

    def get_values_company_by_date(self, company_sym, date, columns = "close"):
        feStock = FE_Stock(company_sym, table_name)
        all_items = feStock.fetch_by_date(date, columns)

        return [float(k[1]) for k in all_items], [k[0] for k in all_items]


    def get_weekly_values_company(self, company_sym, columns="high"):
        feStock = FE_Stock(company_sym, table_name)
        all_items = feStock.fetch_all(columns)

        dates = [k[0] for k in all_items]
        values = [k[1] for k in all_items]

        date_sorted_by_week, values_sorted_by_week = [], []
        temp_d, temp_v = [], []
        week_limits = []

        start_date_db = datetime.datetime.strptime(dates[0], '%Y-%m-%d')
        last_date_db = datetime.datetime.strptime(dates[-1], '%Y-%m-%d')

        temp_date = start_date_db

        week_limits_list = []


        date_sorted_by_week, values_sorted_by_week = [], []

        k = 0
        delta = 0
        flag = 0
        while temp_date < last_date_db:
            temp_date = start_date_db + timedelta(days=delta)
            k_date = datetime.datetime.strptime(dates[k], '%Y-%m-%d')
            if temp_date.weekday() == 0:
                week_limits = [temp_date, temp_date + timedelta(days=5)]
                flag = 1
                if temp_d:
                    date_sorted_by_week += [temp_d]
                    values_sorted_by_week += [temp_v]
                    temp_d = [k_date]
                    temp_v = [values[k]]
                    week_limits = []



            else:

                if not week_limits:

                    if flag:
                        i = 0
                        while (k_date - timedelta(days=i)).weekday() != 0:
                            i += 1

                        week_limits = [k_date - timedelta(i), k_date - timedelta(i) + timedelta(days=5)]

                        continue



            if not temp_date == k_date:
                delta += 1
                continue
            else:

                if week_limits:
                    if k_date >= week_limits[0] and k_date <= week_limits[1]:
                        # print(k_date, week_limits)
                        temp_d += [k_date]
                        temp_v += [values[k]]


            k+=1
            delta += 1
        else:
            if temp_d and temp_v:
                date_sorted_by_week += [temp_d]
                values_sorted_by_week += [temp_v]

        return values_sorted_by_week, date_sorted_by_week


    def get_weekly_stats_company(self, company_sym, columns="high", stats = "max"):
        values_sorted, dates_sorted = self.get_weekly_values_company(company_sym, columns)

        if stats == "max":
            return [max(k) for k in values_sorted], [str(k[-1]).split(" ")[0] for k in dates_sorted]
        elif stats == "min":
            return [min(k) for k in values_sorted], [str(k[-1]) for k in dates_sorted]
        elif stats == "last":
            return [k[-1] for k in values_sorted], [str(k[-1]) for k in dates_sorted]
        else:
            raise Exception("Not a valid stat")


    def get_max_rows(self):
        return self.max_rows


    def get_companies_count(self):
        return len(self.company_list)