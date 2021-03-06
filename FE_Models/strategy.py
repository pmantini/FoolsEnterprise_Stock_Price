from scipy.optimize import linprog
import numpy as np
import os, logging, pickle
from FE_Models.model_DB_Reader import DB_Ops
from FE_Models.optimize import Optimize
import json

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


class RandomSelectionForTwoTimeStepPrediciton(FEStrategy):
    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = ['model']
        self.opt_args = ['output_dir']
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
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Strategy/"
        self.model_input_dir = args["model_input_dir"] if "model_input_dir" in args.keys() else "Output/"
        self.model = args["model"]

        self.eval_dir = args["eval_dir"] if "eval_dir" in args.keys() else self.model_input_dir + "eval_dir/" + self.model + "/"
        self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else self.model_input_dir + "pred_dir/" + self.model + "/"


        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")

        # Evaluation Params
        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30

        logging.info("Test")
        logging.info("Initializeing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        self.companies_count = self.db.get_companies_count()

        self.company_list = self.db.get_list_companies()

        # get prices
        close_prices = self.generate_train_data(column1="close")
        low_prices = self.generate_train_data(column1="close", column2="low")
        high_prices = self.generate_train_data(column1="close", column2="high")

        indices_when_stock_decreased, differnece_when_decreased = np.zeros((close_prices.shape)), np.zeros((close_prices.shape))
        indices_when_stock_increased, differnece_when_increased = np.zeros((close_prices.shape)), np.zeros((close_prices.shape))

        indices_when_stock_decreased[close_prices < 0] = 1
        indices_when_stock_increased[close_prices > 0] = 1

        differnece_when_decreased[indices_when_stock_decreased == 1] = low_prices[indices_when_stock_decreased == 1]
        differnece_when_increased[indices_when_stock_increased == 1] = high_prices[indices_when_stock_increased == 1]

        for k,j in zip(differnece_when_decreased,differnece_when_increased):
            decrease, increase = k[k!=0], j[j != 0]
            self.gaussian_parameters_delta += [{"decrease": [np.mean(decrease), np.std(decrease)],
                                     "increase": [np.mean(increase), np.std(increase)]}]


    def load_model(self, file):
        if str(file).endswith(".json"):

            with open(file, 'rb') as f:
                return pickle.load(f)

        elif str(file).endswith(".npy"):
            return np.load(file)
        else:
            raise Exception("%s does not have a valid file format (json or npy)")

    # def generate_train_data(self, column = "volume"):
    #     company_list = self.db.get_list_companies()
    #     total_companies = self.db.get_companies_count()
    #
    #     max_items = self.db.get_max_rows()
    #
    #     values = np.zeros((total_companies, max_items))
    #
    #     i = 0
    #     for k in company_list:
    #         values_fetch, _ = self.db.get_values_company(company_sym=k, columns=column)
    #         values[i, max_items - len(values_fetch):max_items] = values_fetch
    #         i += 1
    #
    #     total_samples_avail  = max_items
    #
    #     train_samples = values[:,:total_samples_avail]
    #
    #     train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]
    #
    #     #remove zeros, and infs
    #     train_samples[np.isnan(train_samples)] = 0
    #     train_samples[np.isinf(train_samples)] = 0
    #
    #     return train_samples

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

        from_end = 1
        max_items = self.db.get_max_rows() - from_end

        values = np.zeros((total_companies, max_items))

        i = 0
        for k in company_list:
            if i == 0:
                _, dates_fetch = self.db.get_values_company(company_sym=k)
            values_fetch, _ = self.db.get_values_company(company_sym=k, columns=column)
            values_fetch = values_fetch[:-from_end]
            values[i, max_items - len(values_fetch):max_items] = values_fetch
            i += 1


        dates = dates_fetch[-self.days_to_eval:]

        eval_samples = values[:, -self.days_to_eval-1:]
        prices = eval_samples[:, 1:]

        eval_samples = (eval_samples[:, 1:] - eval_samples[:, 0:-1]) / eval_samples[:, 0:-1]

        # remove zeros, and infs
        eval_samples[np.isnan(eval_samples)] = 0
        eval_samples[np.isinf(eval_samples)] = 0

        return eval_samples, dates, prices



    def do_eval(self):
        #load eval file
        eval_data = self.load_model(self.eval_file)

        day = 0
        for k in eval_data:
            days_count = len(eval_data[k]["dates"])

        for day in range(days_count-1):
            temp_predict_data = {}
            temp_close_price = np.zeros(len(self.company_list))
            for k in eval_data:

                temp_predict_data[k] = {"prediction": eval_data[k]["pred"][day:day+2]}


                temp_close_price[self.company_list.index(k)] = eval_data[k]["close"][day]

            stocks, qunts, buy_price, sell_price = self.generate_actions(temp_predict_data, np.array(temp_close_price))


            for k in range(len(stocks)):
                profit = (sell_price[k]-buy_price[k])*qunts[k] if buy_price[k] >= eval_data[self.company_list[stocks[k]]]["low"][day + 1] \
                    and sell_price[k] <= eval_data[self.company_list[stocks[k]]]["high"][day + 2] else 0
                b_status = "bought" if buy_price[k] >= eval_data[self.company_list[stocks[k]]]["low"][day + 1] else "not_bought"
                print(self.company_list[stocks[k]], qunts[k], buy_price[k], eval_data[self.company_list[stocks[k]]]["low"][day + 1], b_status,
                      self.company_list[stocks[k]], sell_price[k], eval_data[self.company_list[stocks[k]]]["high"][day + 2:day + 5],
                      profit)


            day += 1
            exit()


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
            values_fetch = values_fetch[:-1]
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


    def generate_actions(self, pred_data, close_prices):

        list_possible = []
        for k in pred_data:
            if not pred_data[k]["prediction"][0] and pred_data[k]["prediction"][1]:
                list_possible += [k]

        # get prices


        predictions = np.zeros(len(self.company_list))
        drop = np.zeros(len(self.company_list))
        pop = np.zeros(len(self.company_list))


        for i in range(len(self.company_list)):

            while True:
                tempdrop = np.random.normal(self.gaussian_parameters_delta[i]["decrease"][0],
                                            self.gaussian_parameters_delta[i]["decrease"][1])
                # tempdrop = self.gaussian_parameters_delta[i]["decrease"][0]
                if tempdrop < 0 and tempdrop >= self.gaussian_parameters_delta[i]["decrease"][0]:
                    break;

            while True:
                temppop = np.random.normal(self.gaussian_parameters_delta[i]["increase"][0],
                                            self.gaussian_parameters_delta[i]["increase"][1])
                if temppop > 0 and temppop < self.gaussian_parameters_delta[i]["increase"][0]:
                    break;


            drop[i] = tempdrop
            pop[i] = temppop


            if self.company_list[i] in list_possible:
                predictions[i] = -(1+drop[i])*(1+pop[i])
            else:
                predictions[i] = 0

        optimizer = Optimize()
        stock, quantitites = optimizer.random_selection(predictions, close_prices.flatten())

        buy_prices, sell_price = [], []
        for k in stock:
            # print(close_prices[k], close_prices[k] * (1+drop[k]), close_prices[k] * (1+drop[k]) * (1+pop[k]))
            buy_prices += [close_prices[k] * (1+drop[k])]
            sell_price += [close_prices[k] * (1+drop[k]) * (1+pop[k])]

        return stock, quantitites, buy_prices, sell_price



    def do_action(self):
        # load eval file
        pred_data = self.load_model(self.pred_file)
        _, dates, close_prices = self.generate_pred_data(column="close")

        # list_possible = []
        #
        # for k in pred_data:
        #     if not pred_data[k]["prediction"][0] and pred_data[k]["prediction"][1]:
        #         list_possible += [k]

        # print(pred_data)
        # exit()
        stocks, qunts, buy_price, sell_price = self.generate_actions(pred_data, close_prices)

        for k in range(len(stocks)):
            print(self.company_list[stocks[k]], qunts[k], buy_price[k], sell_price[k])

        # print(actions)

class RandomSelectionForTwoTimeStepWeeklyPrediciton(FEStrategy):
    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = ['model']
        self.opt_args = ['output_dir', "resource", "number_of_stocks"]
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
        self.model = args["model"]

        self.eval_dir = args["eval_dir"] if "eval_dir" in args.keys() else self.model_input_dir + "eval_dir/" + self.model + "/"
        self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else self.model_input_dir + "pred_dir/" + self.model + "/"

        self.eval_strategy_dir = args[
            "eval_dir"] if "eval_dir" in args.keys() else self.output_dir + "eval_dir/" + self.name + "/"
        self.pred_strategy_dir = args[
            "pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.resource = float(args["resource"]) if "resource" in args.keys() else 5000
        self.number_of_stocks = int(args["number_of_stocks"]) if "number_of_stocks" in args.keys() else 5


        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")

        self.strategy_eval_file = os.path.join(os.path.dirname(self.eval_strategy_dir), "eval.json")
        self.strategy_pred_file = os.path.join(os.path.dirname(self.pred_strategy_dir), "pred.json")

        # Evaluation Params
        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30

        logging.info("Test")
        logging.info("Initializeing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        self.companies_count = self.db.get_companies_count()

        self.company_list = self.db.get_list_companies()

        # get prices
        close_prices = self.generate_train_data(column1="close")
        low_prices = self.generate_train_data(column1="close", column2="low")
        high_prices = self.generate_train_data(column1="close", column2="high")

        indices_when_stock_decreased, differnece_when_decreased = np.zeros((close_prices.shape)), np.zeros((close_prices.shape))
        indices_when_stock_increased, differnece_when_increased = np.zeros((close_prices.shape)), np.zeros((close_prices.shape))

        indices_when_stock_decreased[close_prices < 0] = 1
        indices_when_stock_increased[close_prices > 0] = 1


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

            with open(file, 'rb') as f:
                return pickle.load(f)

        elif str(file).endswith(".npy"):
            return np.load(file)
        else:
            raise Exception("%s does not have a valid file format (json or npy)")

    # def generate_train_data(self, column = "volume"):
    #     company_list = self.db.get_list_companies()
    #     total_companies = self.db.get_companies_count()
    #
    #     max_items = self.db.get_max_rows()
    #
    #     values = np.zeros((total_companies, max_items))
    #
    #     i = 0
    #     for k in company_list:
    #         values_fetch, _ = self.db.get_values_company(company_sym=k, columns=column)
    #         values[i, max_items - len(values_fetch):max_items] = values_fetch
    #         i += 1
    #
    #     total_samples_avail  = max_items
    #
    #     train_samples = values[:,:total_samples_avail]
    #
    #     train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]
    #
    #     #remove zeros, and infs
    #     train_samples[np.isnan(train_samples)] = 0
    #     train_samples[np.isinf(train_samples)] = 0
    #
    #     return train_samples

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

        from_end = 1
        max_items = self.db.get_max_rows() - from_end

        values = np.zeros((total_companies, max_items))

        i = 0
        for k in company_list:
            if i == 0:
                _, dates_fetch = self.db.get_values_company(company_sym=k)
            values_fetch, _ = self.db.get_values_company(company_sym=k, columns=column)
            values_fetch = values_fetch[:-from_end]
            values[i, max_items - len(values_fetch):max_items] = values_fetch
            i += 1


        dates = dates_fetch[-self.days_to_eval:]

        eval_samples = values[:, -self.days_to_eval-1:]
        prices = eval_samples[:, 1:]

        eval_samples = (eval_samples[:, 1:] - eval_samples[:, 0:-1]) / eval_samples[:, 0:-1]

        # remove zeros, and infs
        eval_samples[np.isnan(eval_samples)] = 0
        eval_samples[np.isinf(eval_samples)] = 0

        return eval_samples, dates, prices



    def do_eval(self):
        #load eval file

        eval_data = self.load_model(self.eval_file)

        day = 0
        for k in eval_data:
            days_count = len(eval_data[k]["dates"])


        for day in range(1,days_count-1):
            temp_predict_data = {}
            temp_close_price = np.zeros(len(self.company_list))
            for k in eval_data:

                temp_predict_data[k] = {"prediction": eval_data[k]["pred"][day:day+2]}


                temp_close_price[self.company_list.index(k)] = eval_data[k]["close"][day-1]


            stocks, qunts, buy_price, sell_price, sell_ratio = self.generate_actions(temp_predict_data, np.array(temp_close_price),
                                                                                     resource=self.resource, number_of_stocks=self.number_of_stocks)

            total = 0
            for k in range(len(stocks)):
                b_status = "bought" if buy_price[k] >= eval_data[self.company_list[stocks[k]]]["low"][
                    day + 1] else "not_bought"
                profit = (sell_price[k]-buy_price[k])*qunts[k] if buy_price[k] >= eval_data[self.company_list[stocks[k]]]["low"][day + 1] \
                                                                  and sell_price[k] <= eval_data[self.company_list[stocks[k]]]["high"][day + 2] else ((eval_data[self.company_list[stocks[k]]]["close"][day+1] - buy_price[k])*qunts[k]
                                                                                                                                                      if b_status == "bought" else 0)



                print(self.company_list[stocks[k]], qunts[k], buy_price[k], eval_data[self.company_list[stocks[k]]]["low"][day + 1], b_status,
                      self.company_list[stocks[k]], sell_price[k], eval_data[self.company_list[stocks[k]]]["high"][day + 2],
                      profit)
                total += profit

            print(total)
            day += 1
            exit()


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
            values_fetch = values_fetch[:-1]
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


    def generate_actions(self, pred_data, close_prices, resource, number_of_stocks):

        list_possible = []


        for k in pred_data:
            if not pred_data[k]["prediction"][0] and pred_data[k]["prediction"][1]:
                list_possible += [k]

        # get prices

        predictions = np.zeros(len(self.company_list))
        drop = np.zeros(len(self.company_list))
        pop = np.zeros(len(self.company_list))


        for i in range(len(self.company_list)):

            while True:
                if self.gaussian_parameters_delta[i]["decrease"][0] and self.gaussian_parameters_delta[i]["decrease"][1]:
                    tempdrop = np.random.normal(self.gaussian_parameters_delta[i]["decrease"][0],
                                                self.gaussian_parameters_delta[i]["decrease"][1])
                else:
                    tempdrop = 0
                    break
                #tempdrop = self.gaussian_parameters_delta[i]["decrease"][0]
                if tempdrop < 0 and tempdrop <= self.gaussian_parameters_delta[i]["decrease"][0]:
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


            if self.company_list[i] in list_possible:
                predictions[i] = -(1+drop[i])*(1+pop[i])
            else:
                predictions[i] = 0

        optimizer = Optimize()
        stock, quantitites = optimizer.random_selection(predictions, close_prices.flatten(), resource=resource, number_of_stocks=number_of_stocks)

        buy_prices, sell_price = [], []
        sell_ratio = []
        for k in stock:
            # print(close_prices[k], close_prices[k] * (1+drop[k]), close_prices[k] * (1+drop[k]) * (1+pop[k]))
            buy_prices += [close_prices[k] * (1+drop[k])]
            sell_price += [close_prices[k] * (1+drop[k]) * (1+pop[k])]
            sell_ratio += [(1+pop[k])]

        return stock, quantitites, buy_prices, sell_price, sell_ratio



    def do_action(self):

        # load pred file
        pred_data = self.load_model(self.pred_file)
        _, dates, close_prices = self.generate_pred_data(column="close")


        stocks, qunts, buy_price, sell_price, sell_ratio = self.generate_actions(pred_data, close_prices,
                                                                                 resource=self.resource, number_of_stocks=self.number_of_stocks)

        actions = {}
        for k in range(len(stocks)):
            print(self.company_list[stocks[k]], qunts[k], buy_price[k], sell_price[k])
            actions[self.company_list[stocks[k]]]={"quantity": qunts[k], "buy": buy_price[k], "sell": sell_price[k],
                                                   "sell_ratio": sell_ratio[k],
                                                   "buy_exp":pred_data[self.company_list[stocks[k]]]["date"][0],
                                                   "sell_exp":pred_data[self.company_list[stocks[k]]]["date"][1]}

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
            pickle.dump(data, outfile)

        outfile.close()

