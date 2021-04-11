# from scipy.optimize import linprog

import numpy as np
import os, logging, pickle
from FE_Models.model_DB_Reader import DB_Ops
from FE_Models.optimize import Optimize
# import json
from setup import mount_folder
import datetime
# import matplotlib.pyplot as plt
# from scipy import stats

logging = logging.getLogger("main")

class FEStrategy:
    def __init__(self, name, req_args, opt_args):
        self.name = name
        self.req_args = req_args
        self.opt_args = opt_args

    def get_args(self):
        return {"required": self.req_args, "optional": self.opt_args}

    def do_eval(self, args):
        pass


    def do_pred(self):
        pass


class RandomSelectionPositiveHubberMarkovDailyWeekly(FEStrategy):
    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = ['models', 'metrics']
        self.opt_args = ['output_dir', "resource", "number_of_stocks", "investment_account", "dropout"]
        FEStrategy.__init__(self, self.name, self.req_args, self.opt_args)

        self.company_list = None
        self.black_list = None
        self.db_folder = None
        self.output_dir = None
        self.model_input_dir = None

        self.db = None

        self.model = None
        self.model_weights_file = None

        self.gaussian_parameters_delta = []


    def do_init(self, args):

        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/Strategy/"
        self.model_input_dir = args["model_input_dir"] if "model_input_dir" in args.keys() else "Output/"
        self.model = [k.strip() for k in args["models"].split(",")]

        self.metric = args["metrics"]

        self.eval_metric_dir = args[
            "eval_dir"] if "eval_dir" in args.keys() else self.model_input_dir + "eval_dir/" + self.metric + "/"
        self.pred_metric_dir = args[
            "metric_dir"] if "metric_dir" in args.keys() else self.model_input_dir + "metric_dir/" + self.metric + "/"

        self.eval_dir = args["eval_dir"] if "eval_dir" in args.keys() else [self.model_input_dir + "eval_dir/" + k + "/" for k in self.model]
        self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else [self.model_input_dir + "pred_dir/" + k + "/" for k in self.model]

        self.eval_strategy_dir = args[
            "eval_dir"] if "eval_dir" in args.keys() else self.output_dir + "eval_dir/" + self.name + "/"
        self.pred_strategy_dir = args[
            "pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.investment_account = args[
            "investment_account"] if "investment_account" in args.keys() else "Alpaca"
        self.input_dir = mount_folder
        self.portfolio_dir = os.path.join(mount_folder, "portfolio")
        self.this_portfolio = os.path.join(self.portfolio_dir, self.investment_account)
        self.account_file = os.path.join(self.this_portfolio, "account.json")

        self.account = None

        if os.path.isfile(self.account_file):

            self.account = self.load_model(self.account_file)

        self.resource = float(args["resource"]) if "resource" in args.keys() else float(self.account["cash"])
        self.number_of_stocks = int(args["number_of_stocks"]) if "number_of_stocks" in args.keys() else 5
        self.dropout = int(args["dropout"]) if "dropout" in args.keys() else 0.25

        self.eval_file = [os.path.join(os.path.dirname(k), "eval.json") for k in self.eval_dir]
        self.pred_file = [os.path.join(os.path.dirname(k), "pred.json") for k in self.pred_dir]

        self.pred_metric_file = os.path.join(os.path.dirname(self.pred_metric_dir), "metric.json")
        self.eval_metric_file = os.path.join(os.path.dirname(self.eval_metric_dir), "eval.json")

        self.strategy_eval_file = os.path.join(os.path.dirname(self.eval_strategy_dir), "eval.json")
        self.strategy_pred_file = os.path.join(os.path.dirname(self.pred_strategy_dir), "pred.json")

        # Evaluation Params
        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        self.companies_count = self.db.get_companies_count()

        self.company_list = self.db.get_list_companies()

        # get prices
        close_prices = self.generate_train_data(column1="high")
        high_prices_change = self.generate_train_data(column1="high")
        low_prices = self.generate_train_data(column1="close", column2="low")
        high_prices = self.generate_train_data(column1="close", column2="high")


        indices_when_stock_decreased, differnece_when_decreased = np.zeros((close_prices.shape)), np.zeros((close_prices.shape))
        indices_when_stock_increased, differnece_when_increased = np.zeros((close_prices.shape)), np.zeros((close_prices.shape))

        indices_when_stock_decreased[high_prices_change < 0] = 1
        indices_when_stock_increased[high_prices_change > 0] = 1

        differnece_when_decreased[indices_when_stock_decreased == 1] = low_prices[indices_when_stock_decreased == 1]
        differnece_when_increased[indices_when_stock_increased == 1] = high_prices[indices_when_stock_increased == 1]

        for k,j in zip(differnece_when_decreased,differnece_when_increased):

            decrease, increase = k[k != 0], j[j != 0]

            if len(decrease) and len(increase):

                self.gaussian_parameters_delta += [{"decrease": [np.mean(decrease), np.std(decrease)],
                                                    "increase": [np.mean(increase), np.std(increase)]}]
            else:
                self.gaussian_parameters_delta += [{"decrease": [0,0],
                                                    "increase": [0,0]}]

    def load_model(self, file):
        if str(file).endswith(".json"):
            print(file)
            with open(file, 'rb') as f:
                return pickle.load(f)

        elif str(file).endswith(".npy"):
            return np.load(file)
        else:
            raise Exception("%s does not have a valid file format (json or npy)")

    def get_prices(self, column1="high"):

        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = self.db.get_max_rows()

        values1 = np.zeros((total_companies, max_items))
        values2 = np.zeros((total_companies, max_items))

        i = 0
        for k in company_list:
            values_fetch1, _ = self.db.get_values_company(company_sym=k, columns=column1)
            values1[i, max_items - len(values_fetch1):max_items] = values_fetch1


            i += 1

        total_samples_avail = max_items

        train_samples1 = values1[:, :total_samples_avail]


        return train_samples1

    def generate_train_data(self, column1 = "volume", column2 = None):
        if not column2:
            column2 = column1
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = self.db.get_max_rows()

        values1 = np.zeros((total_companies, max_items))
        values2 = np.zeros((total_companies, max_items))

        i = 0
        for k in company_list:
            values_fetch1, _ = self.db.get_values_company(company_sym=k, columns=column1)
            values1[i, max_items - len(values_fetch1):max_items] = values_fetch1

            values_fetch2, _ = self.db.get_values_company(company_sym=k, columns=column2)
            values2[i, max_items - len(values_fetch2):max_items] = values_fetch2
            i += 1

        total_samples_avail  = max_items

        train_samples1 = values1[:,:total_samples_avail]
        train_samples2 = values2[:, :total_samples_avail]

        train_samples = (train_samples2[:,1:] - train_samples1[:,0:-1])/train_samples1[:,0:-1]

        #remove zeros, and infs
        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        return train_samples



    def generate_eval_data(self, column = "open"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()


        max_items = self.db.get_max_rows()

        values = np.zeros((total_companies, max_items))
        dates = np.zeros((total_companies, max_items), dtype=object)
        i = 0
        for k in company_list:
            # if i == 0:
            #     _, dates_fetch = self.db.get_values_company(company_sym=k)
            values_fetch, dates_fetch = self.db.get_values_company(company_sym=k, columns=column)
            values_fetch = values_fetch
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            dates[i, max_items - len(dates_fetch):max_items] = dates_fetch

            i += 1


        dates = dates[:, -self.days_to_eval-1:]

        eval_samples = values[:, -self.days_to_eval-1-1:]
        prices = eval_samples[:, 1:]

        eval_samples = (eval_samples[:, 1:] - eval_samples[:, 0:-1]) / eval_samples[:, 0:-1]

        # remove zeros, and infs
        eval_samples[np.isnan(eval_samples)] = 0
        eval_samples[np.isinf(eval_samples)] = 0

        return eval_samples, dates, prices

    def do_eval(self):

        for k in self.eval_file:
            if "weekly" in k.lower():
                eval_data_weekly = self.load_model(k)
            else:
                eval_data_daily = self.load_model(k)

        metrics = self.load_model(self.eval_metric_file)

        close_change, eval_dates, close_data = self.generate_eval_data(column="close")
        _, _ , low_data = self.generate_eval_data(column="low")
        _, _, high_data = self.generate_eval_data(column="high")
        _, _, open_data = self.generate_eval_data(column="open")

        simulations = 50
        average_gain = []
        for sim in range(simulations):
            print("----------------------------------- Simulation %s --------------------------------" % sim)
            cash = self.resource

            holding = dict()
            list_all_hold = set()
            print("Total Resources %s" % self.resource)

            for k in eval_data_daily.keys()[1:]:
                # self.number_of_stock = cash // 1000
                holding[k] = []
                daily_pred =  {"predictions": eval_data_daily[k-1]["pred"]}
                weekly_pred = {"predictions": eval_data_weekly[k-1]["pred"]}
                day_metrics = metrics[k]["metric"]

                today = eval_data_daily[k-1]["dates"][-1]

                stocks, qunts, buy_price, sell_price, sell_ratio, buy_ratio = self.generate_actions(day_metrics,
                                                                                                    daily_pred,
                                                                                                    weekly_pred,
                                                                                                    close_change[:,k-1],
                                                                                                    close_data[:,k-1],
                                                                                                    resource=cash,
                                                                                                    number_of_stocks=self.number_of_stocks,
                                                                                                    dropout=self.dropout)


                for st in range(len(stocks)):
                    if eval_data_daily[k-1]["dates"][stocks[st]] != today:
                        print("Skipping %s" % self.company_list[stocks[st]])
                        continue

                    if buy_price[st] >= open_data[stocks[st], k]:
                        if stocks[st] not in list_all_hold:
                            # print("(open) Buying %s for %s" % (self.company_list[stocks[st]], open_data[stocks[st], k] * qunts[st]))
                            print("(open) Buying %s %s for %s - %s" % (int(qunts[st]),
                            self.company_list[stocks[st]], open_data[stocks[st], k], int(qunts[st])*open_data[stocks[st], k]))

                            cash = cash - open_data[stocks[st], k] * int(qunts[st])
                            holding[k] += [{"stock": stocks[st], "sell_price": open_data[stocks[st], k]*sell_ratio[st], "quants": int(qunts[st]), "status": "h",
                                            "buy_price": open_data[stocks[st], k]}]

                            list_all_hold.add(stocks[st])
                        else:
                            print("Already holding %s" % self.company_list[stocks[st]])

                    elif buy_price[st] >= low_data[stocks[st], k]:
                        if stocks[st] not in list_all_hold:
                            # print("(iDay) Buying %s for %s" % (self.company_list[stocks[st]], buy_price[st] * qunts[st]))
                            print(
                                "(iDay) Buying %s %s for %s - %s" % (int(qunts[st]), self.company_list[stocks[st]], buy_price[st],
                                                                     int(qunts[st])*buy_price[st]))

                            cash = cash - buy_price[st] * int(qunts[st])
                            holding[k] += [{"stock": stocks[st], "sell_price": sell_price[st], "quants": int(qunts[st]), "status": "h",
                                            "buy_price": buy_price[st]}]
                            list_all_hold.add(stocks[st])
                        else:
                            print("Already holding %s" % self.company_list[stocks[st]])


                for day in holding.keys():
                    if k > day:
                        for sell_st in holding[day]:
                            if sell_st["status"] == "h":
                                if sell_st["sell_price"] < open_data[sell_st["stock"], k]:
                                    # print("(open) Selling %s for %s" % (
                                    # self.company_list[sell_st["stock"]], open_data[sell_st["stock"], k] * sell_st["quants"]))
                                    print("(open) Selling %s %s for %s - %s" % (sell_st["quants"],
                                        self.company_list[sell_st["stock"]],
                                        open_data[sell_st["stock"], k], sell_st["quants"]*open_data[sell_st["stock"], k]))
                                    cash += open_data[sell_st["stock"], k] * sell_st["quants"]
                                    sell_st["status"] = 'n'
                                    list_all_hold.remove(sell_st["stock"])
                                elif sell_st["sell_price"] < high_data[sell_st["stock"], k]:
                                    # print("(iDay) Selling %s for %s" % (self.company_list[sell_st["stock"]], sell_st["sell_price"] * sell_st["quants"]))
                                    print("(iDay) Selling %s, %s for %s - %s" % (sell_st["quants"],
                                    self.company_list[sell_st["stock"]], sell_st["sell_price"], sell_st["quants"]*sell_st["sell_price"]))
                                    cash += sell_st["sell_price"] * sell_st["quants"]
                                    sell_st["status"] = 'n'
                                    list_all_hold.remove(sell_st["stock"])

                stock_equity = 0
                total_hold = []
                for day in holding.keys():
                    for holdst in holding[day]:
                        if holdst["status"] == "h":
                            stock_equity += holdst["quants"]*close_data[holdst["stock"], k]
                            position = (close_data[holdst["stock"], k]-holdst["buy_price"])*holdst["quants"]
                            total_hold += ["%s(%s)"  % (self.company_list[holdst["stock"]], position)]



                Equity = cash + stock_equity

                print("day %s, cash: %s, stock equity(%s) %s equity %s" % (k, cash, len(total_hold), stock_equity, Equity))
                print(total_hold)

            average_gain += [Equity]

        print(average_gain)
        print(np.mean(np.array(average_gain) - self.resource))

        return 1

    def generate_pred_data(self, column = "open"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = self.db.get_max_rows()

        values = np.zeros((total_companies, max_items))

        i = 0
        for k in company_list:
            if i == 0:
                _, dates_fetch = self.db.get_values_company(company_sym=k)
            values_fetch, _ = self.db.get_values_company(company_sym=k, columns=column)
            # values_fetch = values_fetch[:-1]
            values[i, max_items - len(values_fetch):max_items] = values_fetch
            i += 1

        dates = dates_fetch[-1:]
        prices = values[:, -1:]

        pred_samples = values[:, -2:]

        pred_samples = (pred_samples[:, 1:] - pred_samples[:, 0:-1]) / pred_samples[:, 0:-1]

        # remove zeros, and infs
        pred_samples[np.isnan(pred_samples)] = 0
        pred_samples[np.isinf(pred_samples)] = 0

        # print(pred_samples)

        return pred_samples, dates, prices


    def generate_actions(self, hubber_parameters, pred_data_daily, pred_data_weekly, close_changes, close_prices, resource, number_of_stocks, dropout = 0.25):

        list_possible = []

        for ind, k in enumerate(hubber_parameters):
            if k[0] > 0.02 and pred_data_daily["predictions"][ind] == 0 and pred_data_weekly["predictions"][ind] == 2:
                list_possible += [ind]


        # filtered_close_changes = np.ones(close_changes.shape)*100
        # filtered_close_changes[list_possible] = close_changes[list_possible]

        predictions = np.zeros(len(list_possible))
        drop = np.zeros(len(list_possible))
        pop = np.zeros(len(list_possible))

        for ind, i in enumerate(list_possible):
            while True:
                if self.gaussian_parameters_delta[i]["decrease"][0] and self.gaussian_parameters_delta[i]["decrease"][1]:
                    tempdrop = np.random.normal(self.gaussian_parameters_delta[i]["decrease"][0],
                                                self.gaussian_parameters_delta[i]["decrease"][1])
                else:
                    tempdrop = 0
                    break

                if tempdrop < 0 and tempdrop >= self.gaussian_parameters_delta[i]["decrease"][0]:
                    break;

            while True:
                if self.gaussian_parameters_delta[i]["increase"][0] and self.gaussian_parameters_delta[i]["increase"][1]:
                    temppop = np.random.normal(self.gaussian_parameters_delta[i]["increase"][0],
                                               self.gaussian_parameters_delta[i]["increase"][1])
                else:
                    temppop = 0
                    break
                # temppop = self.gaussian_parameters_delta[i]["increase"][0]

                if temppop > 0 and temppop < self.gaussian_parameters_delta[i]["increase"][0]:
                    break;


            drop[ind] = tempdrop
            pop[ind] = temppop


            predictions[ind] = (1+drop[ind]) - (1+drop[ind])*(1+pop[ind])


        optimizer = Optimize()
        # stock, quantitites = optimizer.random_selection(filtered_close_changes.flatten(), close_prices.flatten(),
        #                                                 resource=resource, number_of_stocks=number_of_stocks,
        #                                                 dropout=dropout)
        stock, quantitites, reordered_indices = optimizer.random_selection(predictions, list_possible, close_prices.flatten(),
                                                            resource=resource, number_of_stocks=number_of_stocks,
                                                            dropout=dropout)

        buy_prices, sell_price = [], []
        sell_ratio, buy_ratio = [], []

        for k, reindex in zip(stock, reordered_indices):

            buy_prices += [close_prices[k] * (1 + drop[reindex])]
            sell_price += [close_prices[k] * (1 + drop[reindex]) * (1 + pop[reindex])]
            sell_ratio += [(1 + pop[reindex])]
            buy_ratio += [(1 + drop[reindex])]

        return stock, quantitites, buy_prices, sell_price, sell_ratio, buy_ratio

    def do_action(self):

        for k in self.pred_file:
            if "weekly" in k.lower():
                pred_data_weekly = self.load_model(k)
            else:
                pred_data_daily = self.load_model(k)

        metrics = self.load_model(self.pred_metric_file)

        close_changes, dates, close_prices = self.generate_pred_data(column="close")

        stocks, qunts, buy_price, sell_price, sell_ratio, buy_ratio = self.generate_actions(metrics, pred_data_daily, pred_data_weekly, close_changes, close_prices,
                                                                                 resource=self.resource, number_of_stocks=self.number_of_stocks, dropout=self.dropout)

        actions = {}

        def get_next_day_delta(this_date):
            if this_date.weekday() == 4:
                return this_date + datetime.timedelta(days = 3)
            elif this_date.weekday() == 5:
                return this_date + datetime.timedelta(days = 2)
            else:
                return this_date + datetime.timedelta(days=1)

        for k in range(len(stocks)):
            print(stocks[k], self.company_list[stocks[k]], qunts[k], buy_price[k], sell_price[k])

            current_date = datetime.datetime.strptime(pred_data_daily["dates"][stocks[k]], '%Y-%m-%d')

            this_buy_exp = get_next_day_delta(current_date)
            this_sell_exp = current_date + datetime.timedelta(days=7)

            this_buy_exp = datetime.datetime.strftime(this_buy_exp, '%Y-%m-%d')
            this_sell_exp = datetime.datetime.strftime(this_sell_exp, '%Y-%m-%d')

            actions[self.company_list[stocks[k]]]={"quantity": qunts[k], "buy": buy_price[k], "sell": sell_price[k],
                                                   "sell_ratio": sell_ratio[k],
                                                   "buy_ratio": buy_ratio[k],
                                                   "buy_exp": this_buy_exp,
                                                   "sell_exp": this_sell_exp}

        print(actions)
        self.save_pred_output(actions)

    def save_pred_output(self, data):
        print("save", "---------------------")
        print(self.strategy_pred_file)
        print(self.pred_strategy_dir)
        try:
            os.makedirs(self.pred_strategy_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.strategy_pred_file)
        else:
            logging.info("Successfully created the directory %s " % self.strategy_pred_file)

        logging.info("Writing strategy output to %s", self.strategy_pred_file)

        with open(self.strategy_pred_file, 'wb') as outfile:
            pickle.dump(data, outfile, protocol=2)

        outfile.close()


class RandomPennySelectionPositiveHubberMarkovDailyWeekly(FEStrategy):
    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = ['models', 'metrics']
        self.opt_args = ['output_dir', "resource", "number_of_stocks", "investment_account", "dropout"]
        FEStrategy.__init__(self, self.name, self.req_args, self.opt_args)

        self.company_list = None
        self.black_list = None
        self.db_folder = None
        self.output_dir = None
        self.model_input_dir = None

        self.db = None

        self.model = None
        self.model_weights_file = None

        self.gaussian_parameters_delta = []


    def do_init(self, args):

        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/Strategy/"
        self.model_input_dir = args["model_input_dir"] if "model_input_dir" in args.keys() else "Output/"
        self.model = [k.strip() for k in args["models"].split(",")]

        self.metric = args["metrics"]

        self.eval_metric_dir = args[
            "eval_dir"] if "eval_dir" in args.keys() else self.model_input_dir + "eval_dir/" + self.metric + "/"
        self.pred_metric_dir = args[
            "metric_dir"] if "metric_dir" in args.keys() else self.model_input_dir + "metric_dir/" + self.metric + "/"

        self.eval_dir = args["eval_dir"] if "eval_dir" in args.keys() else [self.model_input_dir + "eval_dir/" + k + "/" for k in self.model]
        self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else [self.model_input_dir + "pred_dir/" + k + "/" for k in self.model]

        self.eval_strategy_dir = args[
            "eval_dir"] if "eval_dir" in args.keys() else self.output_dir + "eval_dir/" + self.name + "/"
        self.pred_strategy_dir = args[
            "pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.investment_account = args[
            "investment_account"] if "investment_account" in args.keys() else "Alpaca"
        self.input_dir = mount_folder
        self.portfolio_dir = os.path.join(mount_folder, "portfolio")
        self.this_portfolio = os.path.join(self.portfolio_dir, self.investment_account)
        self.account_file = os.path.join(self.this_portfolio, "account.json")

        self.account = None

        if os.path.isfile(self.account_file):

            self.account = self.load_model(self.account_file)

        self.resource = float(args["resource"]) if "resource" in args.keys() else float(self.account["cash"])
        self.number_of_stocks = int(args["number_of_stocks"]) if "number_of_stocks" in args.keys() else 5
        self.dropout = int(args["dropout"]) if "dropout" in args.keys() else 0.25

        self.eval_file = [os.path.join(os.path.dirname(k), "eval.json") for k in self.eval_dir]
        self.pred_file = [os.path.join(os.path.dirname(k), "pred.json") for k in self.pred_dir]

        self.pred_metric_file = os.path.join(os.path.dirname(self.pred_metric_dir), "metric.json")
        self.eval_metric_file = os.path.join(os.path.dirname(self.eval_metric_dir), "eval.json")

        self.strategy_eval_file = os.path.join(os.path.dirname(self.eval_strategy_dir), "eval.json")
        self.strategy_pred_file = os.path.join(os.path.dirname(self.pred_strategy_dir), "pred.json")

        # Evaluation Params
        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        self.companies_count = self.db.get_companies_count()

        self.company_list = self.db.get_list_companies()

        # get prices
        close_prices = self.generate_train_data(column1="high")
        high_prices_change = self.generate_train_data(column1="high")
        low_prices = self.generate_train_data(column1="close", column2="low")
        high_prices = self.generate_train_data(column1="close", column2="high")


        indices_when_stock_decreased, differnece_when_decreased = np.zeros((close_prices.shape)), np.zeros((close_prices.shape))
        indices_when_stock_increased, differnece_when_increased = np.zeros((close_prices.shape)), np.zeros((close_prices.shape))

        indices_when_stock_decreased[high_prices_change < 0] = 1
        indices_when_stock_increased[high_prices_change > 0] = 1

        differnece_when_decreased[indices_when_stock_decreased == 1] = low_prices[indices_when_stock_decreased == 1]
        differnece_when_increased[indices_when_stock_increased == 1] = high_prices[indices_when_stock_increased == 1]

        for k,j in zip(differnece_when_decreased,differnece_when_increased):

            decrease, increase = k[k != 0], j[j != 0]

            if len(decrease) and len(increase):

                self.gaussian_parameters_delta += [{"decrease": [np.mean(decrease), np.std(decrease)],
                                                    "increase": [np.mean(increase), np.std(increase)]}]
            else:
                self.gaussian_parameters_delta += [{"decrease": [0,0],
                                                    "increase": [0,0]}]

    def load_model(self, file):
        if str(file).endswith(".json"):
            print(file)
            with open(file, 'rb') as f:
                return pickle.load(f)

        elif str(file).endswith(".npy"):
            return np.load(file)
        else:
            raise Exception("%s does not have a valid file format (json or npy)")

    def get_prices(self, column1="high"):

        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = self.db.get_max_rows()

        values1 = np.zeros((total_companies, max_items))
        values2 = np.zeros((total_companies, max_items))

        i = 0
        for k in company_list:
            values_fetch1, _ = self.db.get_values_company(company_sym=k, columns=column1)
            values1[i, max_items - len(values_fetch1):max_items] = values_fetch1


            i += 1

        total_samples_avail = max_items

        train_samples1 = values1[:, :total_samples_avail]


        return train_samples1

    def generate_train_data(self, column1 = "volume", column2 = None):
        if not column2:
            column2 = column1
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = self.db.get_max_rows()

        values1 = np.zeros((total_companies, max_items))
        values2 = np.zeros((total_companies, max_items))

        i = 0
        for k in company_list:
            values_fetch1, _ = self.db.get_values_company(company_sym=k, columns=column1)
            values1[i, max_items - len(values_fetch1):max_items] = values_fetch1

            values_fetch2, _ = self.db.get_values_company(company_sym=k, columns=column2)
            values2[i, max_items - len(values_fetch2):max_items] = values_fetch2
            i += 1

        total_samples_avail  = max_items

        train_samples1 = values1[:,:total_samples_avail]
        train_samples2 = values2[:, :total_samples_avail]

        train_samples = (train_samples2[:,1:] - train_samples1[:,0:-1])/train_samples1[:,0:-1]

        #remove zeros, and infs
        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        return train_samples



    def generate_eval_data(self, column = "open"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()


        max_items = self.db.get_max_rows()

        values = np.zeros((total_companies, max_items))
        dates = np.zeros((total_companies, max_items), dtype=object)
        i = 0
        for k in company_list:
            # if i == 0:
            #     _, dates_fetch = self.db.get_values_company(company_sym=k)
            values_fetch, dates_fetch = self.db.get_values_company(company_sym=k, columns=column)
            values_fetch = values_fetch
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            dates[i, max_items - len(dates_fetch):max_items] = dates_fetch

            i += 1


        dates = dates[:, -self.days_to_eval-1:]

        eval_samples = values[:, -self.days_to_eval-1-1:]
        prices = eval_samples[:, 1:]

        eval_samples = (eval_samples[:, 1:] - eval_samples[:, 0:-1]) / eval_samples[:, 0:-1]

        # remove zeros, and infs
        eval_samples[np.isnan(eval_samples)] = 0
        eval_samples[np.isinf(eval_samples)] = 0

        return eval_samples, dates, prices

    def do_eval(self):

        for k in self.eval_file:
            if "weekly" in k.lower():
                eval_data_weekly = self.load_model(k)
            else:
                eval_data_daily = self.load_model(k)

        metrics = self.load_model(self.eval_metric_file)

        close_change, eval_dates, close_data = self.generate_eval_data(column="close")
        _, _ , low_data = self.generate_eval_data(column="low")
        _, _, high_data = self.generate_eval_data(column="high")
        _, _, open_data = self.generate_eval_data(column="open")

        simulations = 50
        average_gain = []
        for sim in range(simulations):
            print("----------------------------------- Simulation %s --------------------------------" % sim)
            cash = self.resource

            holding = dict()
            list_all_hold = set()
            print("Total Resources %s" % self.resource)
            for k in eval_data_daily.keys()[1:]:
                # self.number_of_stock = cash // 1000
                holding[k] = []
                daily_pred =  {"predictions": eval_data_daily[k-1]["pred"]}
                weekly_pred = {"predictions": eval_data_weekly[k-1]["pred"]}
                day_metrics = metrics[k]["metric"]

                today = eval_data_daily[k-1]["dates"][-1]

                stocks, qunts, buy_price, sell_price, sell_ratio, buy_ratio = self.generate_actions(day_metrics,
                                                                                                    daily_pred,
                                                                                                    weekly_pred,
                                                                                                    close_change[:,k-1],
                                                                                                    close_data[:,k-1],
                                                                                                    resource=cash,
                                                                                                    number_of_stocks=self.number_of_stocks,
                                                                                                    dropout=self.dropout)


                for st in range(len(stocks)):
                    if eval_data_daily[k-1]["dates"][stocks[st]] != today:
                        print("Skipping %s" % self.company_list[stocks[st]])
                        continue
                    if buy_price[st] >= open_data[stocks[st], k]:
                        if stocks[st] not in list_all_hold:
                            print("Buying %s for %s" % (self.company_list[stocks[st]], buy_price[st] * qunts[st]))

                            cash = cash - open_data[stocks[st], k] * int(qunts[st])
                            holding[k] += [{"stock": stocks[st], "sell_price": sell_price[st], "quants": int(qunts[st]), "status": "h",
                                            "buy_price": open_data[stocks[st], k]}]
                            list_all_hold.add(stocks[st])
                        else:
                            print("Already holding %s" % self.company_list[stocks[st]])

                    elif buy_price[st] >= low_data[stocks[st], k]:
                        if stocks[st] not in list_all_hold:
                            print("Buying %s for %s" % (self.company_list[stocks[st]], buy_price[st] * qunts[st]))

                            cash = cash - buy_price[st] * int(qunts[st])
                            holding[k] += [{"stock": stocks[st], "sell_price": sell_price[st], "quants": int(qunts[st]), "status": "h",
                                            "buy_price": buy_price[st]}]
                            list_all_hold.add(stocks[st])
                        else:
                            print("Already holding %s" % self.company_list[stocks[st]])


                for day in holding.keys():
                    if k > day:
                        for sell_st in holding[day]:
                            if sell_st["status"] == "h":
                                if sell_st["sell_price"] < open_data[sell_st["stock"], k]:
                                    print("Selling %s for %s" % (
                                    self.company_list[sell_st["stock"]], open_data[sell_st["stock"], k] * sell_st["quants"]))
                                    cash += open_data[sell_st["stock"], k] * sell_st["quants"]
                                    sell_st["status"] = 'n'
                                    list_all_hold.remove(sell_st["stock"])
                                elif sell_st["sell_price"] < high_data[sell_st["stock"], k]:
                                    print("Selling %s for %s" % (self.company_list[sell_st["stock"]], sell_st["sell_price"] * sell_st["quants"]))
                                    cash += sell_st["sell_price"] * sell_st["quants"]
                                    sell_st["status"] = 'n'
                                    list_all_hold.remove(sell_st["stock"])

                stock_equity = 0
                total_hold = []
                for day in holding.keys():
                    for holdst in holding[day]:
                        if holdst["status"] == "h":
                            stock_equity += holdst["quants"]*close_data[holdst["stock"], k]
                            position = (close_data[holdst["stock"], k]-holdst["buy_price"])*holdst["quants"]
                            total_hold += ["%s(%s)"  % (self.company_list[holdst["stock"]], position)]



                Equity = cash + stock_equity

                print("day %s, cash: %s, stock equity(%s) %s equity %s" % (k, cash, len(total_hold), stock_equity, Equity))
                print(total_hold)

            average_gain += [Equity]

        print(average_gain)
        print(np.mean(np.array(average_gain) - self.resource))

        return 1

    def generate_pred_data(self, column = "open"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = self.db.get_max_rows()

        values = np.zeros((total_companies, max_items))

        i = 0
        for k in company_list:
            if i == 0:
                _, dates_fetch = self.db.get_values_company(company_sym=k)
            values_fetch, _ = self.db.get_values_company(company_sym=k, columns=column)
            # values_fetch = values_fetch[:-1]
            values[i, max_items - len(values_fetch):max_items] = values_fetch
            i += 1

        dates = dates_fetch[-1:]
        prices = values[:, -1:]

        pred_samples = values[:, -2:]

        pred_samples = (pred_samples[:, 1:] - pred_samples[:, 0:-1]) / pred_samples[:, 0:-1]

        # remove zeros, and infs
        pred_samples[np.isnan(pred_samples)] = 0
        pred_samples[np.isinf(pred_samples)] = 0

        # print(pred_samples)

        return pred_samples, dates, prices


    def generate_actions(self, hubber_parameters, pred_data_daily, pred_data_weekly, close_changes, close_prices, resource, number_of_stocks, dropout = 0.25):

        list_possible = []

        for ind, k in enumerate(hubber_parameters):
            if k[0] > 0.0 and pred_data_daily["predictions"][ind] == 0 and pred_data_weekly["predictions"][ind] == 2:
                list_possible += [ind]


        filtered_close_changes = np.ones(close_changes.shape)*100
        filtered_close_changes[list_possible] = close_changes[list_possible]

        # predictions = np.zeros(len(list_possible))
        drop = np.zeros(len(self.company_list))
        pop = np.zeros(len(self.company_list))

        for i in list_possible:
            while True:
                if self.gaussian_parameters_delta[i]["decrease"][0] and self.gaussian_parameters_delta[i]["decrease"][1]:
                    tempdrop = np.random.normal(self.gaussian_parameters_delta[i]["decrease"][0],
                                                self.gaussian_parameters_delta[i]["decrease"][1])
                else:
                    tempdrop = 0
                    break

                if tempdrop < 0 and tempdrop >= self.gaussian_parameters_delta[i]["decrease"][0]:
                    break;

            while True:
                if self.gaussian_parameters_delta[i]["increase"][0] and self.gaussian_parameters_delta[i]["increase"][1]:
                    temppop = np.random.normal(self.gaussian_parameters_delta[i]["increase"][0],
                                               self.gaussian_parameters_delta[i]["increase"][1])
                else:
                    temppop = 0
                    break
                # temppop = self.gaussian_parameters_delta[i]["increase"][0]

                if temppop > 0 and temppop < self.gaussian_parameters_delta[i]["increase"][0]:
                    break;


            drop[i] = tempdrop
            pop[i] = temppop

            # predictions[ind] = (1 + drop[ind]) - (1 + drop[ind]) * (1 + pop[ind])

        optimizer = Optimize()
        stock, quantitites = optimizer.random_selection_penny(filtered_close_changes.flatten(), close_prices.flatten(),
                                                              resource=resource, number_of_stocks=number_of_stocks,
                                                              dropout=dropout)

        buy_prices, sell_price = [], []
        sell_ratio, buy_ratio = [], []

        for k in stock:
            buy_prices += [close_prices[k] * (1 + drop[k])]
            sell_price += [close_prices[k] * (1 + drop[k]) * (1 + pop[k])]
            sell_ratio += [(1 + pop[k])]
            buy_ratio += [(1 + drop[k])]

        return stock, quantitites, buy_prices, sell_price, sell_ratio, buy_ratio

    def do_action(self):

        for k in self.pred_file:
            if "weekly" in k.lower():
                pred_data_weekly = self.load_model(k)
            else:
                pred_data_daily = self.load_model(k)

        metrics = self.load_model(self.pred_metric_file)

        close_changes, dates, close_prices = self.generate_pred_data(column="close")

        stocks, qunts, buy_price, sell_price, sell_ratio, buy_ratio = self.generate_actions(metrics, pred_data_daily, pred_data_weekly, close_changes, close_prices,
                                                                                 resource=self.resource, number_of_stocks=self.number_of_stocks, dropout=self.dropout)

        actions = {}

        def get_next_day_delta(this_date):
            if this_date.weekday() == 4:
                return this_date + datetime.timedelta(days = 3)
            elif this_date.weekday() == 5:
                return this_date + datetime.timedelta(days = 2)
            else:
                return this_date + datetime.timedelta(days=1)

        for k in range(len(stocks)):
            print(stocks[k], self.company_list[stocks[k]], qunts[k], buy_price[k], sell_price[k])

            current_date = datetime.datetime.strptime(pred_data_daily["dates"][stocks[k]], '%Y-%m-%d')

            this_buy_exp = get_next_day_delta(current_date)
            this_sell_exp = current_date + datetime.timedelta(days=7)

            this_buy_exp = datetime.datetime.strftime(this_buy_exp, '%Y-%m-%d')
            this_sell_exp = datetime.datetime.strftime(this_sell_exp, '%Y-%m-%d')

            actions[self.company_list[stocks[k]]]={"quantity": qunts[k], "buy": buy_price[k], "sell": sell_price[k],
                                                   "sell_ratio": sell_ratio[k],
                                                   "buy_ratio": buy_ratio[k],
                                                   "buy_exp": this_buy_exp,
                                                   "sell_exp": this_sell_exp}

        print(actions)
        self.save_pred_output(actions)

    def save_pred_output(self, data):
        print("save", "---------------------")
        print(self.strategy_pred_file)
        print(self.pred_strategy_dir)
        try:
            os.makedirs(self.pred_strategy_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.strategy_pred_file)
        else:
            logging.info("Successfully created the directory %s " % self.strategy_pred_file)

        logging.info("Writing strategy output to %s", self.strategy_pred_file)

        with open(self.strategy_pred_file, 'wb') as outfile:
            pickle.dump(data, outfile, protocol=2)

        outfile.close()
