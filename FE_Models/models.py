from keras.layers import Dense, Flatten, Dropout, BatchNormalization, LSTM, Bidirectional, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.callbacks import TensorBoard
from FE_Models.model_DB_Reader import DB_Ops
from keras.optimizers import Adam, Nadam, SGD
from keras.models import model_from_json
import random
from setup import log_file
import pickle
import datetime

from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.utils import to_categorical

import numpy as np

import logging
import os
import shutil
from itertools import dropwhile


logging = logging.getLogger("main")


class FEModel:
    def __init__(self, name, req_args, opt_args):
        self.name = name
        self.req_args = req_args
        self.opt_args = opt_args

    def get_args(self):
        return {"required": self.req_args, "optional": self.opt_args}

    def do_train_and_eval(self, args):
        pass


    def do_eval(self):
        pass

    def do_pred(self):
        pass


class markov_o1_c2(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = []
        self.opt_args = ['ouput_dir', "days_to_eval", "model_dir", 'pred_dir']
        FEModel.__init__(self, self.name, self.req_args, self.opt_args)

        self.company_list = None
        self.black_list = None
        self.db_folder = None
        self.output_dir = None

        self.db = None

        self.input_shape = None
        self.points_model = None

        self.model_dir = None

        self.model_file = None
        self.model_weights_file = None


    def do_init(self, args):
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"

        self.model_dir = args["model_dir"] if "model_dir" in args.keys() else self.output_dir+"training_dir/"+self.name+"/"
        self.eval_dir = args["eval_dir"] if "eval_dir" in args.keys() else self.output_dir+"eval_dir/"+self.name+"/"
        self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.model_file = os.path.join(os.path.dirname(self.model_dir), "model.npy")
        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")


        #Evaluation Params
        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30

        logging.info("Test")
        logging.info("Initializeing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        self.companies_count = self.db.get_companies_count()

        self.points_model = self.model_init()



    def model_init(self):

        return 0

    def save_model(self, transision_matrix):
        # Create Folder
        try:
            os.makedirs(self.model_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        np.save(self.model_file, transision_matrix)
        logging.info("Saving transision matrix to %s", self.model_file)


    def load_model(self):
        return np.load(self.model_file)

    def generate_train_data(self, column = "volume"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = self.db.get_max_rows()

        values = np.zeros((total_companies, max_items))

        i = 0
        for k in company_list:
            values_fetch, _ = self.db.get_values_company(company_sym=k, columns=column)
            values[i, max_items - len(values_fetch):max_items] = values_fetch
            i += 1

        total_samples_avail  = max_items

        train_samples = values[:,:total_samples_avail]

        train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]

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



    def get_class(self, mat, labels={0: 0.00}):
        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)
        for k in range(len(labels)):

            if k == len(labels) - 1:
                continue

            matclasses[np.logical_and(mat > labels[k], mat <= labels[k + 1])] = k + 1
        else:

            matclasses[mat > labels[len(labels) - 1]] = len(labels)

        return matclasses


    def do_train(self):

        #get data
        train_data = self.generate_train_data(column="close")

        number_of_classes = 2


        # train_data = get_class(train_data, labels={0: 0})
        train_data = self.get_class(train_data)

        transision_matrix = np.zeros((self.companies_count, number_of_classes, number_of_classes))

        i = 0
        for k in train_data:

            for tminus1, t in zip(k[:-1], k[1:]):
                transision_matrix[i, tminus1, t] += 1

            i += 1


        self.save_model(transision_matrix)
        return

    def do_eval(self):
        #### Computes overall average accuracy, per stock accuracy
        # evaluation
        transision_matrix = self.load_model()

        eval_data, dates, prices = self.generate_eval_data(column="close")

        eval_classes = self.get_class(eval_data)


        def pred_all_stocks(eval_data, eval_classes):
            eval_pred = np.zeros(eval_classes.shape, dtype=np.int)
            i = 0
            for eval_element in eval_data:
                j = 0
                for state in eval_classes[i]:
                    initial_state = state
                    if j == 0:
                        eval_pred[i, j] = initial_state

                    this_trans = transision_matrix[i]
                    eval_pred[i, j] = self.predict_next_state(initial_state, this_trans)

                    j += 1
                i += 1
            return eval_pred

        eval_pred = pred_all_stocks(eval_data, eval_classes)

        output = dict()
        from sklearn.metrics import accuracy_score

        y_true = eval_classes.flatten()
        y_pred = eval_pred.flatten()
        # for g, p in zip(y_true, y_pred):
        #     print(g, p)

        print("Overall Accuracy:", accuracy_score(y_true, y_pred))
        output["overall"] = {"accuracy": accuracy_score(y_true, y_pred)}


        self.save_eval_output(output)



    def predict_next_state(self, initial_state, transision_matrix):

        this_trans = transision_matrix
        this_trans /= this_trans.sum(axis=1).reshape(-1, 1)
        trans_cumsum = np.cumsum(this_trans, axis=1)
        randomvalue = np.random.random()
        class_iter = 0


        for pty in trans_cumsum[initial_state]:
            if randomvalue < pty:
                break

            class_iter += 1

        return class_iter


    def do_pred(self):
        #### Computes overall average accuracy, per stock accuracy
        # evaluation
        transision_matrix = self.load_model()

        pred_data, dates, prices = self.generate_pred_data(column="close")
        _, _, prices = self.generate_pred_data(column="open")

        pred_classes = self.get_class(pred_data)
        # pred_pred = np.zeros(pred_classes.shape, dtype=np.int)
        pred_pred = dict()

        i = 0
        company_list = self.db.get_list_companies()
        last_date = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
        predict_date = last_date + datetime.timedelta(days=1)
        for pred_element in pred_classes:
            comp = company_list[i]
            pred_pred[comp] = {"date": str(predict_date).split(" ")[0]}
            pred_pred[comp]["prediction"] = self.predict_next_state(pred_element[0], transision_matrix[i])

            i += 1


        # strategy_obj = Strategy()
        # resource = 500
        # choices, qunatities = strategy_obj.linear_problem(pred_classes[:,0], prices.T[0], resource=resource)
        #
        # print(predict_date, [company_list[k] for k in choices], qunatities, [pred_classes[k] for k in choices])
        # pred_pred["scenario"] = {"stocks": [company_list[k] for k in choices], "qantities": qunatities}
        # pred_pred["view"] = {"stocks": [company_list[k] for k in choices], "qantities": qunatities}

        print(pred_pred)
        self.save_pred_output(pred_pred)


    def save_eval_output(self, data):

        try:
            os.makedirs(self.eval_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.eval_dir)
        else:
            logging.info("Successfully created the directory %s " % self.eval_dir)

        logging.info("Writing evaluation output to %s", self.eval_file)

        with open(self.eval_file, 'wb') as outfile:
            pickle.dump(data, outfile)

        outfile.close()

    def save_pred_output(self, data):

        try:
            os.makedirs(self.pred_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.pred_dir)
        else:
            logging.info("Successfully created the directory %s " % self.pred_dir)

        logging.info("Writing evaluation output to %s", self.pred_file)

        with open(self.pred_file, 'wb') as outfile:
            pickle.dump(data, outfile)

        outfile.close()

class markov_o2_c2(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = []
        self.opt_args = ['ouput_dir', "days_to_eval", "model_dir", 'pred_dir']
        FEModel.__init__(self, self.name, self.req_args, self.opt_args)

        self.company_list = None
        self.black_list = None
        self.db_folder = None
        self.output_dir = None

        self.db = None

        self.input_shape = None
        self.points_model = None

        self.model_dir = None

        self.model_file = None
        self.model_weights_file = None


    def do_init(self, args):
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"

        self.model_dir = args["model_dir"] if "model_dir" in args.keys() else self.output_dir+"training_dir/"+self.name+"/"
        self.eval_dir = args["eval_dir"] if "eval_dir" in args.keys() else self.output_dir+"eval_dir/"+self.name+"/"
        self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.model_file = os.path.join(os.path.dirname(self.model_dir), "model.npy")
        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")


        #Evaluation Params
        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30

        logging.info("Test")
        logging.info("Initializeing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        self.companies_count = self.db.get_companies_count()

        self.points_model = self.model_init()



    def model_init(self):

        return 0

    def save_model(self, transision_matrix):
        # Create Folder
        try:
            os.makedirs(self.model_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        np.save(self.model_file, transision_matrix)
        logging.info("Saving transision matrix to %s", self.model_file)


    def load_model(self):
        return np.load(self.model_file)

    def generate_train_data(self, column = "volume"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = self.db.get_max_rows()

        values = np.zeros((total_companies, max_items))

        i = 0
        for k in company_list:
            values_fetch, _ = self.db.get_values_company(company_sym=k, columns=column)
            values[i, max_items - len(values_fetch):max_items] = values_fetch
            i += 1

        total_samples_avail  = max_items

        train_samples = values[:,:total_samples_avail]

        train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]

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

        dates = dates_fetch[-2:]
        prices = values[:, -2:]

        pred_samples = values[:, -3:]

        pred_samples = (pred_samples[:, 1:] - pred_samples[:, 0:-1]) / pred_samples[:, 0:-1]

        # remove zeros, and infs
        pred_samples[np.isnan(pred_samples)] = 0
        pred_samples[np.isinf(pred_samples)] = 0

        # print(pred_samples)

        return pred_samples, dates, prices



    def get_class(self, mat, labels={0: 0.0}):

        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)
        for k in range(len(labels)):

            if k == len(labels) - 1:
                continue

            matclasses[np.logical_and(mat > labels[k], mat <= labels[k + 1])] = k + 1
        else:

            matclasses[mat > labels[len(labels) - 1]] = len(labels)

        matclasses2 = np.zeros((matclasses.shape[0], matclasses.shape[1] - 1), dtype=np.int)

        labels_print = range(2)
        for i in labels_print:
            for j in labels_print:
                print(i, j, i * 2 + j)

        k = 0
        for stock in matclasses:
            two_days_classes = [(1 + len(labels)) * i + j for i, j in zip(stock[:-1], stock[1:])]
            matclasses2[k] = two_days_classes
            k += 1

        return matclasses2


    def do_train(self):

        #get data
        train_data = self.generate_train_data(column="close")

        number_of_classes = 4


        # train_data = get_class(train_data, labels={0: 0})
        train_data = self.get_class(train_data)

        transision_matrix = np.zeros((self.companies_count, number_of_classes, number_of_classes))

        i = 0
        for k in train_data:

            for tminus1, t in zip(k[:-1], k[1:]):
                transision_matrix[i, tminus1, t] += 1

            i += 1


        self.save_model(transision_matrix)
        return

    def do_eval(self):
        #### Computes overall average accuracy, per stock accuracy
        # evaluation
        transision_matrix = self.load_model()

        eval_data, dates, prices_close = self.generate_eval_data(column="close")
        _, _, prices_low = self.generate_eval_data(column="low")
        _, _, prices_high = self.generate_eval_data(column="high")

        eval_classes = self.get_class(eval_data)

        eval_classes_new = []
        for k in eval_classes:


            lm = [k[0] // 2]
            lm += [r % 2 for r in k]

            eval_classes_new += [lm]


        def pred_all_stocks(eval_data, eval_classes):
            eval_pred = np.zeros(eval_classes.shape, dtype=np.int)
            i = 0
            for eval_element in eval_data:
                j = 0
                for state in eval_classes[i]:
                    initial_state = state
                    if j == 0:
                        eval_pred[i, j] = initial_state

                    this_trans = transision_matrix[i]
                    eval_pred[i, j] = self.predict_next_state(initial_state, this_trans)

                    j += 1
                i += 1

            return eval_pred

        eval_pred = pred_all_stocks(eval_data, eval_classes)

        eval_pred_new = []
        for k in eval_pred:
            lm = [k[0] // 2]
            lm += [r % 2 for r in k]

            eval_pred_new += [lm]
        eval_classes = np.array(eval_classes_new)
        eval_pred = np.array(eval_pred_new)


        output = dict()
        from sklearn.metrics import accuracy_score

        confidence = []

        eval_iter = 0
        company_list = self.db.get_list_companies()


        for y_true, y_pred in zip(eval_classes, eval_pred):
            comp = company_list[eval_iter]
            output[comp] = {"accuracy": accuracy_score(y_true, y_pred), "gt": y_true, "pred":y_pred,
                            "close": prices_close[eval_iter],
                            "high": prices_high[eval_iter],
                            "low": prices_low[eval_iter],
                            "dates": dates}


            confidence += [output[comp]["accuracy"]]

            eval_iter += 1

        y_true = eval_classes.flatten()
        y_pred = eval_pred.flatten()

        # for g, p, a in zip(y_true, y_pred, eval_data.flatten()):
        #     if g == p:
        #         print(g,p,a)
        print("overall accuracy: ", accuracy_score(y_true, y_pred))
        # output["overall"] = {"accuracy": accuracy_score(y_true, y_pred)}
        # for k in output:
        #     print(k)
        #     print(k, output[k]["pred"])
        #     print(k,output[k]["gt"])

        self.save_eval_output(output)


    def predict_next_state(self, initial_state, transision_matrix):

        this_trans = transision_matrix
        this_trans /= this_trans.sum(axis=1).reshape(-1, 1)
        trans_cumsum = np.cumsum(this_trans, axis=1)
        randomvalue = np.random.random()
        class_iter = 0


        for pty in trans_cumsum[initial_state]:
            if randomvalue < pty:
                break

            class_iter += 1

        return class_iter



    def do_pred(self):
        #### Computes overall average accuracy, per stock accuracy
        # evaluation
        transision_matrix = self.load_model()

        pred_data, dates, prices = self.generate_pred_data(column="close")

        _, _, prices = self.generate_pred_data(column="open")

        pred_classes = self.get_class(pred_data)

        # pred_pred = np.zeros(pred_classes.shape, dtype=np.int)
        pred_pred = dict()

        i = 0
        company_list = self.db.get_list_companies()
        last_date = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
        predict_date = last_date + datetime.timedelta(days=1)
        for pred_element in pred_classes:
            comp = company_list[i]
            pred_pred[comp] = {"date": str(predict_date).split(" ")[0]}
            pred_pred[comp]["prediction"] = self.predict_next_state(pred_element[0], transision_matrix[i])

            i += 1


        # strategy_obj = Strategy()
        # resource = 500
        # choices, qunatities = strategy_obj.linear_problem(pred_classes[:,0], prices.T[0], resource=resource)
        #
        # print(predict_date, [company_list[k] for k in choices], qunatities, [pred_classes[k] for k in choices])
        # pred_pred["scenario"] = {"stocks": [company_list[k] for k in choices], "qantities": qunatities}
        # pred_pred["view"] = {"stocks": [company_list[k] for k in choices], "qantities": qunatities}


        self.save_pred_output(pred_pred)

    def do_pred(self, time_step = 2):
        #### Computes overall average accuracy, per stock accuracy
        # evaluation
        transision_matrix = self.load_model()

        pred_data, dates, prices = self.generate_pred_data(column="close")
        _, _, prices = self.generate_pred_data(column="open")

        pred_classes = self.get_class(pred_data)


        # pred_pred = np.zeros(pred_classes.shape, dtype=np.int)
        pred_pred = dict()
        for t in range(time_step):
            tem_pred = []
            i = 0
            company_list = self.db.get_list_companies()
            last_date = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
            predict_date = last_date + datetime.timedelta(days=t+1)
            for pred_element in pred_classes:
                comp = company_list[i]
                if comp not in pred_pred.keys():
                    pred_pred[comp] = {"date": [str(predict_date).split(" ")[0]]}
                    pred_pred[comp]["prediction"] = [self.predict_next_state(pred_element[0], transision_matrix[i])]

                else:
                    pred_pred[comp]['date'] += [str(predict_date).split(" ")[0]]
                    pred_pred[comp]["prediction"] += [self.predict_next_state(pred_element[0], transision_matrix[i])]
                tem_pred += [pred_pred[comp]["prediction"]]
                i += 1
            pred_classes = tem_pred

        for each_class in pred_pred:
            test = []

            lm = [pred_pred[each_class]["prediction"][0] // 2]
            lm += [r % 2 for r in pred_pred[each_class]["prediction"]]

            test += lm
            pred_pred[each_class]["prediction"] = test[1:]

        # strategy_obj = Strategy()
        # resource = 500
        # choices, qunatities = strategy_obj.linear_problem(pred_classes[:,0], prices.T[0], resource=resource)
        #
        # print(predict_date, [company_list[k] for k in choices], qunatities, [pred_classes[k] for k in choices])
        # pred_pred["scenario"] = {"stocks": [company_list[k] for k in choices], "qantities": qunatities}
        # pred_pred["view"] = {"stocks": [company_list[k] for k in choices], "qantities": qunatities}

        print(pred_pred)
        self.save_pred_output(pred_pred)


    def save_eval_output(self, data):

        try:
            os.makedirs(self.eval_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.eval_dir)
        else:
            logging.info("Successfully created the directory %s " % self.eval_dir)

        logging.info("Writing evaluation output to %s", self.eval_file)

        with open(self.eval_file, 'wb') as outfile:
            pickle.dump(data, outfile)

        outfile.close()

    def save_pred_output(self, data):

        try:
            os.makedirs(self.pred_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.pred_dir)
        else:
            logging.info("Successfully created the directory %s " % self.pred_dir)

        logging.info("Writing evaluation output to %s", self.pred_file)

        with open(self.pred_file, 'wb') as outfile:
            pickle.dump(data, outfile)

        outfile.close()


class lstm_w(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = []
        self.opt_args = ['ouput_dir', "days_to_eval", "model_dir", 'pred_dir', 'n_steps', 'n_features']
        FEModel.__init__(self, self.name, self.req_args, self.opt_args)

        self.company_list = None
        self.black_list = None
        self.db_folder = None
        self.output_dir = None

        self.db = None

        self.input_shape = None
        self.points_model = None

        self.model_dir = None

        self.model_file = None
        self.model_weights_file = None

        self.n_steps = None
        self.n_features = None

    def do_init(self, args):
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"

        self.model_dir = args["model_dir"] if "model_dir" in args.keys() else self.output_dir+"training_dir/"+self.name+"/"
        self.eval_dir = args["eval_dir"] if "eval_dir" in args.keys() else self.output_dir+"eval_dir/"+self.name+"/"
        self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.n_steps = args[
            "n_steps"] if "n_steps" in args.keys() else 20
        self.n_features = args[
            "n_features"] if "n_features" in args.keys() else 1

        self.model_file = os.path.join(os.path.dirname(self.model_dir), "model.h5")
        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")

        self.intial_model = self.model_file.replace("model.h5", "initilaization" + ".h5")
        #Evaluation Params
        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30

        logging.info("Test")
        logging.info("Initializeing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        self.companies_count = self.db.get_companies_count()

        self.lstm_model = self.model_init()

    def model_init(self):

        # define model
        model = Sequential()
        #
        model.add(Bidirectional(LSTM(150, activation='relu'), input_shape=(self.n_steps, self.n_features)))
        # model.add(LSTM(100, return_sequences=True, activation='relu'))
        # model.add(LSTM(150, return_sequences=True, activation='relu'))
        # model.add(LSTM(100, return_sequences=True, activation='relu'))
        # model.add(LSTM(50, activation='relu'))

        # model = Sequential()
        # model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
        #                           input_shape=(None, 2, self.n_features)))
        # model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        # model.add(TimeDistributed(Flatten()))
        # model.add(LSTM(50, activation='relu'))
        # model.add(Dense(1))

        model.add(Dense(3))


        model.compile(optimizer='adam', loss='mse')

        return model

    def generate_train_data(self, column = "volume", stats = "max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        # max_items = self.db.get_max_rows()
        max_items = 0
        values_array, dates_array = [], []
        for k in company_list:
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]

            # _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        # i = 0
        for i in range(len(company_list)):
            if i == 0:
                # _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
                #max_items = len(dates_fetch)
                values = np.zeros((total_companies, max_items))
            # values_fetch, _ = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch
            # i += 1

        weeks_to_eval = self.days_to_eval//7 if 1 else 0

        total_samples_avail  = max_items - weeks_to_eval

        train_samples = values[:,:total_samples_avail]

        train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]

        #remove zeros, and infs
        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        return train_samples

    # split a univariate sequence
    def split_sequence(self, sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern

            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def do_train(self):
        # get data
        train_data = self.generate_train_data(column="high")

        self.save_model(self.intial_model)
        # import matplotlib.pyplot as plt
        stock_no = 0
        # avg_deg_1, avg_deg_2 = 0, 0
        fitting_parmas = []
        from sklearn.metrics import accuracy_score
        global_1, global_1_y = [], []
        global_2, global_2_y = [], []
        global_pred = []
        global_t = []
        for k in train_data:

            self.lstm_model.load_weights(self.intial_model)
            this_model = self.model_file.replace("model.h5", str(stock_no)+".h5")

            k = np.array(list(dropwhile(lambda x: x==0, k)))

            k[k <= -0.01] = -1
            k = np.array([0 if r >= -0.01 and r <= 0.01 else r for r in k])
            k[k > 0.01] = 1
            k += 1


            X, y_ = self.split_sequence(k, self.n_steps)
            y = to_categorical(y_, 3)
            if not X.shape[0]:
                self.save_model(this_model)
                stock_no += 1
                continue

            X = X.reshape((X.shape[0], X.shape[1], self.n_features))

            # n_seq = 2
            # n_steps = 2
            # X = X.reshape((X.shape[0], n_seq, n_steps, self.n_features))


            self.lstm_model.fit(X[:-2], y[:-2], epochs=200, verbose=0)

            self.save_model(this_model)


            pred_1 = np.argmax(self.lstm_model.predict(np.expand_dims(X[-2], axis=0)))
            X[-1][-1] = pred_1
            pred_2 = np.argmax(self.lstm_model.predict(np.expand_dims(X[-1], axis=0)))

            pred = [pred_1, pred_2]
            y = y_[-2:].flatten()
            global_1 += [pred_1]
            global_1_y += [y[0]]

            global_2 += [pred_2]
            global_2_y += [y[1]]

            global_t += [y]
            global_pred += [pred]
            # pred[pred >= 0.5] = 1
            # pred[pred < 0.5] = 0

            # print(pred, y)
            print("Finised Training: ", stock_no, "accuracy =", accuracy_score(y, pred), y, pred, "------", "Running: ", accuracy_score(np.array(global_t).flatten(), np.array(global_pred).flatten()),
                  "1: ", accuracy_score(np.array(global_1_y).flatten(), np.array(global_1).flatten()), "2: ", accuracy_score(np.array(global_2_y).flatten(), np.array(global_2).flatten()) )

            # for u, v in zip(pred, y):
            #     u[u >= 0.5] = 1
            #     u[u < 0.5] = 0
            #     u = u.flatten()

                # print("p", u, v)
            # exit()
            #
            #
            # slope_1, intercept_1 = np.polyfit(y[:,0], pred[:,0], 1)
            # slope_2, intercept_2 = np.polyfit(y[:, 1], pred[:, 1], 1)
            #
            # fitting_parmas += [[slope_1, intercept_1, slope_2, intercept_2]]
            # print("Finised Training: ", stock_no)
            stock_no += 1

        # pred_fitting = os.path.join(self.model_dir, "fit.npy")
        # np.save(pred_fitting, fitting_parmas)
            # print(np.rad2deg(np.arctan(slope_1)))
            # print(np.arctan(slope_2))
            # dif_p, dif_a = [], []
            # for p, a in zip(pred, y):
            #     dif_p += [p[1] - p[0]]
            #     dif_a += [a[1] - a[0]]
            # #
            # plt.plot(dif_p, dif_a, ".")
            # day_1, day_2 = [], []
            # for p, a in zip(pred, y):
            #     day_1 += [p[0], a[0]]
            #     day_2 += [p[1], a[1]]
            # avg_deg_1 += 90-np.rad2deg(np.arctan(slope_1))
            # avg_deg_2 += 90-np.rad2deg(np.arctan(slope_2))
            #
            # plt.plot(y[:,0] * slope_1 + intercept_1, y[:,0], 'g', label=90-np.rad2deg(np.arctan(slope_1)))
            # plt.plot(y[:,1] * slope_2 + intercept_2, y[:,1], 'r', label=90-np.rad2deg(np.arctan(slope_2)))
            # plt.plot(pred[:,0], y[:,0],"g.")
            # plt.plot(pred[:,1], y[:,1],"r.")
            #
            # print(stock_no, 90-np.rad2deg(np.arctan(slope_1)), 90-np.rad2deg(np.arctan(slope_2)), avg_deg_1/stock_no, avg_deg_2/stock_no)
        # # print(avg_deg_1/stock_no)
        # # print(avg_deg_2 / stock_no)
        #
        #     plt.legend()
        #     plt.show()

    def generate_eval_data(self, column = "open", stats = "max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        from_end = 1
        weeks_to_eval = self.days_to_eval//7

        max_items = 0
        values_array, dates_array = [], []
        for k in company_list:
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]

            # _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        # i = 0
        for i in range(len(company_list)):
            if i == 0:
                # _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
                # max_items = len(dates_fetch)
                values = np.zeros((total_companies, max_items))
            # values_fetch, _ = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_fetch = values_array[i][:-from_end]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

        dates = dates_array[0][-(weeks_to_eval+ self.n_steps)-2:]

        eval_samples = values[:, -(weeks_to_eval+ self.n_steps)-2:]
        prices = eval_samples[:, 1:]

        eval_samples = (eval_samples[:, 1:] - eval_samples[:, 0:-1]) / eval_samples[:, 0:-1]

        # remove zeros, and infs
        eval_samples[np.isnan(eval_samples)] = 0
        eval_samples[np.isinf(eval_samples)] = 0

        return eval_samples, dates, prices

    def do_eval(self):
        #### Computes overall average accuracy, per stock accuracy
        # evaluation
        eval_data, dates, prices_close = self.generate_eval_data(column="high")
        stock_no = 0

        def get_classes(vals):
            vals = vals.flatten()
            classes = np.zeros(vals.shape[0])
            classes[vals >= 0.5] = 1

            return classes

        from sklearn.metrics import accuracy_score

        y_true, y_pred = [], []
        for k in eval_data:

            this_model = self.model_file.replace("model.h5", str(stock_no) + ".h5")
            self.lstm_model.load_weights(this_model)

            k[k <= 0] = 0
            k[k > 0] = 1

            X, y = self.split_sequence(k, self.n_steps)

            X = X.reshape((X.shape[0], X.shape[1], self.n_features))

            predictions = self.lstm_model.predict(X)

            classes_pred = get_classes(predictions)
            classes_actual = y.flatten()
            print(classes_pred)
            print(classes_actual)

            y_true.extend(classes_actual[2:])
            y_pred.extend(classes_pred[2:])

            stock_no += 1

        print(accuracy_score(y_true, y_pred))





    def save_model(self, model_weights):
        # Create Folder
        try:
            os.makedirs(self.model_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        self.lstm_model.save_weights(model_weights)
        logging.info("Saving model weights to %s", self.model_file)


class markov_o2_c2_w(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = []
        self.opt_args = ['ouput_dir', "days_to_eval", "model_dir", 'pred_dir']
        FEModel.__init__(self, self.name, self.req_args, self.opt_args)

        self.company_list = None
        self.black_list = None
        self.db_folder = None
        self.output_dir = None

        self.db = None

        self.input_shape = None
        self.points_model = None

        self.model_dir = None

        self.model_file = None
        self.model_weights_file = None


    def do_init(self, args):
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"

        self.model_dir = args["model_dir"] if "model_dir" in args.keys() else self.output_dir+"training_dir/"+self.name+"/"
        self.eval_dir = args["eval_dir"] if "eval_dir" in args.keys() else self.output_dir+"eval_dir/"+self.name+"/"
        self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.model_file = os.path.join(os.path.dirname(self.model_dir), "model.npy")
        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")


        #Evaluation Params
        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30

        logging.info("Test")
        logging.info("Initializeing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        self.companies_count = self.db.get_companies_count()

        self.points_model = self.model_init()




    def model_init(self):

        return 0

    def save_model(self, transision_matrix):
        # Create Folder
        try:
            os.makedirs(self.model_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        np.save(self.model_file, transision_matrix)
        logging.info("Saving transision matrix to %s", self.model_file)


    def load_model(self):
        return np.load(self.model_file)




    def generate_train_data(self, column = "volume", stats = "max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        # max_items = self.db.get_max_rows()
        max_items = 0
        values_array, dates_array = [], []
        for k in company_list:
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]

            # _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        # i = 0
        for i in range(len(company_list)):
            if i == 0:
                # _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
                #max_items = len(dates_fetch)
                values = np.zeros((total_companies, max_items))
            # values_fetch, _ = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch
            # i += 1


        total_samples_avail  = max_items

        train_samples = values[:,:total_samples_avail]

        train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]

        #remove zeros, and infs
        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        return train_samples

    def generate_eval_data(self, column = "open", stats = "max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        from_end = 1
        weeks_to_eval = self.days_to_eval//7

        max_items = 0
        values_array, dates_array = [], []
        for k in company_list:
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]

            # _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        # i = 0
        for i in range(len(company_list)):
            if i == 0:
                # _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
                # max_items = len(dates_fetch)
                values = np.zeros((total_companies, max_items))
            # values_fetch, _ = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_fetch = values_array[i][:-from_end]
            values[i, max_items - len(values_fetch):max_items] = values_fetch



        # max_items = 0
        # for k in company_list:
        #     _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
        #     if max_items < len(dates_fetch) - 1:
        #         max_items = len(dates_fetch) - 1
        #
        #
        # i = 0
        # for k in company_list:
        #
        #     if i == 0:
        #         _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
        #         # max_items = len(dates_fetch) - 1
        #
        #         values = np.zeros((total_companies, max_items))
        #
        #     values_fetch, _ = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
        #     values_fetch = values_fetch[:-from_end]
        #
        #
        #     values[i, max_items - len(values_fetch):max_items] = values_fetch
        #     i += 1


        dates = dates_array[0][-weeks_to_eval-2:]
        # dates = dates_fetch[-weeks_to_eval - 2:]

        eval_samples = values[:, -weeks_to_eval-2:]
        prices = eval_samples[:, 1:]

        eval_samples = (eval_samples[:, 1:] - eval_samples[:, 0:-1]) / eval_samples[:, 0:-1]

        # remove zeros, and infs
        eval_samples[np.isnan(eval_samples)] = 0
        eval_samples[np.isinf(eval_samples)] = 0

        return eval_samples, dates, prices

    def generate_pred_data(self, column = "open"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        # max_items = 0
        # for k in company_list:
        #     _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k)
        #     if max_items < len(dates_fetch) - 1:
        #         max_items = len(dates_fetch) - 1
        #
        # i = 0
        # for k in company_list:
        #     if i == 0:
        #         _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k)
        #         # max_items = len(dates_fetch) - 1
        #
        #         values = np.zeros((total_companies, max_items))
        #
        #     values_fetch, _ = self.db.get_weekly_stats_company(company_sym=k, columns=column)
        #     values_fetch = values_fetch[:-1]
        #     values[i, max_items - len(values_fetch):max_items] = values_fetch
        #     i += 1

        max_items = 0
        values_array, dates_array = [], []
        for k in company_list:
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column)
            values_array += [v_temp]
            dates_array += [d_temp]

            # _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        # i = 0
        for i in range(len(company_list)):
            if i == 0:
                # _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
                # max_items = len(dates_fetch)
                values = np.zeros((total_companies, max_items))
            # values_fetch, _ = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_fetch = values_array[i][:-1]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

        dates = dates_array[0][-2:]
        prices = values[:, -2:]

        pred_samples = values[:, -3:]

        pred_samples = (pred_samples[:, 1:] - pred_samples[:, 0:-1]) / pred_samples[:, 0:-1]

        # remove zeros, and infs
        pred_samples[np.isnan(pred_samples)] = 0
        pred_samples[np.isinf(pred_samples)] = 0

        # print(pred_samples)

        return pred_samples, dates, prices



    def get_class(self, mat, labels={0: 0.0}):

        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)
        for k in range(len(labels)):

            if k == len(labels) - 1:
                continue

            matclasses[np.logical_and(mat > labels[k], mat <= labels[k + 1])] = k + 1
        else:

            matclasses[mat > labels[len(labels) - 1]] = len(labels)

        matclasses2 = np.zeros((matclasses.shape[0], matclasses.shape[1] - 1), dtype=np.int)

        # labels_print = range(2)
        # for i in labels_print:
        #     for j in labels_print:
        #         print(i, j, i * 2 + j)

        k = 0
        for stock in matclasses:

            two_days_classes = [(1 + len(labels)) * i + j for i, j in zip(stock[:-1], stock[1:])]
            matclasses2[k] = two_days_classes
            k += 1

        return matclasses2


    def do_train(self):

        #get data
        train_data = self.generate_train_data(column="high")

        number_of_classes = 4


        # train_data = get_class(train_data, labels={0: 0})
        train_data = self.get_class(train_data)

        transision_matrix = np.zeros((self.companies_count, number_of_classes, number_of_classes))

        i = 0
        for k in train_data:

            for tminus1, t in zip(k[:-1], k[1:]):
                transision_matrix[i, tminus1, t] += 1

            i += 1


        self.save_model(transision_matrix)
        return

    def do_eval(self):
        #### Computes overall average accuracy, per stock accuracy
        # evaluation

        transision_matrix = self.load_model()

        eval_data, dates, prices_close = self.generate_eval_data(column="high")

        _, _, prices_low = self.generate_eval_data(column="low", stats="min")
        _, _, prices_high = self.generate_eval_data(column="high")

        _, _, prices_c = self.generate_eval_data(column="close", stats="last")

        eval_data = eval_data[:,1:]

        eval_classes = self.get_class(eval_data)

        eval_classes_new = []
        for k in eval_classes:


            lm = [k[0] // 2]
            lm += [r % 2 for r in k]

            eval_classes_new += [lm]


        def pred_all_stocks(eval_data, eval_classes):
            eval_pred = np.zeros(eval_classes.shape, dtype=np.int)
            i = 0
            for eval_element in eval_data:
                j = 0
                for state in eval_classes[i]:
                    initial_state = state
                    if j == 0:
                        eval_pred[i, j] = initial_state

                    this_trans = transision_matrix[i]
                    eval_pred[i, j] = self.predict_next_state(initial_state, this_trans)

                    j += 1
                i += 1

            return eval_pred

        eval_pred = pred_all_stocks(eval_data, eval_classes)

        eval_pred_new = []
        for k in eval_pred:
            lm = [k[0] // 2]
            lm += [r % 2 for r in k]

            eval_pred_new += [lm]
        eval_classes = np.array(eval_classes_new)
        eval_pred = np.array(eval_pred_new)


        output = dict()
        from sklearn.metrics import accuracy_score

        confidence = []

        eval_iter = 0
        company_list = self.db.get_list_companies()


        for y_true, y_pred in zip(eval_classes, eval_pred):
            comp = company_list[eval_iter]
            output[comp] = {"accuracy": accuracy_score(y_true, y_pred), "gt": y_true, "pred":y_pred,
                            "close": prices_c[eval_iter],
                            "high": prices_high[eval_iter],
                            "low": prices_low[eval_iter],
                            "dates": dates}


            confidence += [output[comp]["accuracy"]]

            eval_iter += 1

        y_true = eval_classes.flatten()
        y_pred = eval_pred.flatten()
        print(y_true)
        # print(y_pred)
        # for g, p, a in zip(y_true, y_pred, eval_data.flatten()):
        #     if g == p:
        #         print(g,p,a)
        print("overall accuracy: ", accuracy_score(y_true, y_pred))
        # print(confidence)
        # output["overall"] = {"accuracy": accuracy_score(y_true, y_pred)}
        # for k in output:
        #     print(k)
        #     print(k, output[k]["pred"])
        #     print(k,output[k]["gt"])
        # print(output)
        self.save_eval_output(output)


    def predict_next_state(self, initial_state, transision_matrix):

        this_trans = transision_matrix
        this_trans /= this_trans.sum(axis=1).reshape(-1, 1)
        trans_cumsum = np.cumsum(this_trans, axis=1)
        randomvalue = np.random.random()
        class_iter = 0


        for pty in trans_cumsum[initial_state]:
            if randomvalue < pty:
                break

            class_iter += 1

        return class_iter



    # def do_pred(self):
    #     #### Computes overall average accuracy, per stock accuracy
    #     # evaluation
    #     transision_matrix = self.load_model()
    #
    #     pred_data, dates, prices = self.generate_pred_data(column="close")
    #
    #     _, _, prices = self.generate_pred_data(column="open")
    #
    #     pred_classes = self.get_class(pred_data)
    #
    #     # pred_pred = np.zeros(pred_classes.shape, dtype=np.int)
    #     pred_pred = dict()
    #
    #     i = 0
    #     company_list = self.db.get_list_companies()
    #     last_date = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
    #     predict_date = last_date + datetime.timedelta(days=1)
    #     for pred_element in pred_classes:
    #         comp = company_list[i]
    #         pred_pred[comp] = {"date": str(predict_date).split(" ")[0]}
    #         pred_pred[comp]["prediction"] = self.predict_next_state(pred_element[0], transision_matrix[i])
    #
    #         i += 1
    #
    #
    #     # strategy_obj = Strategy()
    #     # resource = 500
    #     # choices, qunatities = strategy_obj.linear_problem(pred_classes[:,0], prices.T[0], resource=resource)
    #     #
    #     # print(predict_date, [company_list[k] for k in choices], qunatities, [pred_classes[k] for k in choices])
    #     # pred_pred["scenario"] = {"stocks": [company_list[k] for k in choices], "qantities": qunatities}
    #     # pred_pred["view"] = {"stocks": [company_list[k] for k in choices], "qantities": qunatities}
    #
    #
    #     self.save_pred_output(pred_pred)

    def do_pred(self, time_step = 2):
        #### Computes overall average accuracy, per stock accuracy
        # evaluation
        transision_matrix = self.load_model()
        print(transision_matrix.shape)
        import time
        start = time.time()
        pred_data, dates, prices = self.generate_pred_data(column="high")

        print(pred_data.shape)
        pred_classes = self.get_class(pred_data)
        print(pred_classes.shape)
        # _, dates, prices = self.generate_pred_data(column="close")
        end = time.time()
        print(end - start)



        # pred_pred = np.zeros(pred_classes.shape, dtype=np.int)
        pred_pred = dict()
        for t in range(time_step):
            tem_pred = []
            i = 0
            company_list = self.db.get_list_companies()

            last_date = datetime.datetime.strptime(dates[1], "%Y-%m-%d")

            predict_date = last_date + datetime.timedelta(days=7*(t+1))

            for pred_element in pred_classes:

                comp = company_list[i]
                if comp not in pred_pred.keys():
                    pred_pred[comp] = {"date": [str(predict_date).split(" ")[0]]}
                    pred_pred[comp]["prediction"] = [self.predict_next_state(pred_element[0], transision_matrix[i])]

                else:
                    pred_pred[comp]['date'] += [str(predict_date).split(" ")[0]]
                    pred_pred[comp]["prediction"] += [self.predict_next_state(pred_element[0], transision_matrix[i])]
                tem_pred += [pred_pred[comp]["prediction"]]
                i += 1
            pred_classes = tem_pred

        for each_class in pred_pred:
            test = []

            lm = [pred_pred[each_class]["prediction"][0] // 2]
            lm += [r % 2 for r in pred_pred[each_class]["prediction"]]

            test += lm
            pred_pred[each_class]["prediction"] = test[1:]

        # strategy_obj = Strategy()
        # resource = 500
        # choices, qunatities = strategy_obj.linear_problem(pred_classes[:,0], prices.T[0], resource=resource)
        #
        # print(predict_date, [company_list[k] for k in choices], qunatities, [pred_classes[k] for k in choices])
        # pred_pred["scenario"] = {"stocks": [company_list[k] for k in choices], "qantities": qunatities}
        # pred_pred["view"] = {"stocks": [company_list[k] for k in choices], "qantities": qunatities}

        print(pred_pred)
        self.save_pred_output(pred_pred)


    def save_eval_output(self, data):

        try:
            os.makedirs(self.eval_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.eval_dir)
        else:
            logging.info("Successfully created the directory %s " % self.eval_dir)

        logging.info("Writing evaluation output to %s", self.eval_file)

        with open(self.eval_file, 'wb') as outfile:
            pickle.dump(data, outfile)

        outfile.close()

    def save_pred_output(self, data):

        try:
            os.makedirs(self.pred_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.pred_dir)
        else:
            logging.info("Successfully created the directory %s " % self.pred_dir)

        logging.info("Writing evaluation output to %s", self.pred_file)

        with open(self.pred_file, 'wb') as outfile:
            pickle.dump(data, outfile)

        outfile.close()


class markov_o2_c3_w(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = []
        self.opt_args = ['ouput_dir', "days_to_eval", "model_dir", 'pred_dir']
        FEModel.__init__(self, self.name, self.req_args, self.opt_args)

        self.company_list = None
        self.black_list = None
        self.db_folder = None
        self.output_dir = None

        self.db = None

        self.input_shape = None
        self.points_model = None

        self.model_dir = None

        self.model_file = None
        self.model_weights_file = None


    def do_init(self, args):
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"

        self.model_dir = args["model_dir"] if "model_dir" in args.keys() else self.output_dir+"training_dir/"+self.name+"/"
        self.eval_dir = args["eval_dir"] if "eval_dir" in args.keys() else self.output_dir+"eval_dir/"+self.name+"/"
        self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.model_file = os.path.join(os.path.dirname(self.model_dir), "model.npy")
        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")


        #Evaluation Params
        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30

        logging.info("Test")
        logging.info("Initializeing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        self.companies_count = self.db.get_companies_count()

        self.points_model = self.model_init()




    def model_init(self):

        return 0

    def save_model(self, transision_matrix):
        # Create Folder
        try:
            os.makedirs(self.model_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        np.save(self.model_file, transision_matrix)
        logging.info("Saving transision matrix to %s", self.model_file)


    def load_model(self):
        return np.load(self.model_file)




    def generate_train_data(self, column = "volume", stats = "max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        # max_items = self.db.get_max_rows()
        max_items = 0
        values_array, dates_array = [], []
        for k in company_list:
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]

            # _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        # i = 0
        for i in range(len(company_list)):
            if i == 0:
                # _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
                #max_items = len(dates_fetch)
                values = np.zeros((total_companies, max_items))
            # values_fetch, _ = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch
            # i += 1


        total_samples_avail  = max_items

        train_samples = values[:,:total_samples_avail]

        train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]

        #remove zeros, and infs
        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        return train_samples

    def generate_eval_data(self, column = "open", stats = "max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        from_end = 1
        weeks_to_eval = self.days_to_eval//7

        max_items = 0
        values_array, dates_array = [], []
        for k in company_list:
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]

            # _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        # i = 0
        for i in range(len(company_list)):
            if i == 0:
                # _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
                # max_items = len(dates_fetch)
                values = np.zeros((total_companies, max_items))
            # values_fetch, _ = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_fetch = values_array[i][:-from_end]
            values[i, max_items - len(values_fetch):max_items] = values_fetch



        # max_items = 0
        # for k in company_list:
        #     _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
        #     if max_items < len(dates_fetch) - 1:
        #         max_items = len(dates_fetch) - 1
        #
        #
        # i = 0
        # for k in company_list:
        #
        #     if i == 0:
        #         _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
        #         # max_items = len(dates_fetch) - 1
        #
        #         values = np.zeros((total_companies, max_items))
        #
        #     values_fetch, _ = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
        #     values_fetch = values_fetch[:-from_end]
        #
        #
        #     values[i, max_items - len(values_fetch):max_items] = values_fetch
        #     i += 1


        dates = dates_array[0][-weeks_to_eval-2:]
        # dates = dates_fetch[-weeks_to_eval - 2:]

        eval_samples = values[:, -weeks_to_eval-2:]
        prices = eval_samples[:, 1:]

        eval_samples = (eval_samples[:, 1:] - eval_samples[:, 0:-1]) / eval_samples[:, 0:-1]

        # remove zeros, and infs
        eval_samples[np.isnan(eval_samples)] = 0
        eval_samples[np.isinf(eval_samples)] = 0

        return eval_samples, dates, prices

    def generate_pred_data(self, column = "open"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        # max_items = 0
        # for k in company_list:
        #     _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k)
        #     if max_items < len(dates_fetch) - 1:
        #         max_items = len(dates_fetch) - 1
        #
        # i = 0
        # for k in company_list:
        #     if i == 0:
        #         _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k)
        #         # max_items = len(dates_fetch) - 1
        #
        #         values = np.zeros((total_companies, max_items))
        #
        #     values_fetch, _ = self.db.get_weekly_stats_company(company_sym=k, columns=column)
        #     values_fetch = values_fetch[:-1]
        #     values[i, max_items - len(values_fetch):max_items] = values_fetch
        #     i += 1

        max_items = 0
        values_array, dates_array = [], []
        for k in company_list:
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column)
            values_array += [v_temp]
            dates_array += [d_temp]

            # _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        # i = 0
        for i in range(len(company_list)):
            if i == 0:
                # _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
                # max_items = len(dates_fetch)
                values = np.zeros((total_companies, max_items))
            # values_fetch, _ = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_fetch = values_array[i][:-1]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

        dates = dates_array[0][-2:]
        prices = values[:, -2:]

        pred_samples = values[:, -3:]

        pred_samples = (pred_samples[:, 1:] - pred_samples[:, 0:-1]) / pred_samples[:, 0:-1]

        # remove zeros, and infs
        pred_samples[np.isnan(pred_samples)] = 0
        pred_samples[np.isinf(pred_samples)] = 0

        # print(pred_samples)

        return pred_samples, dates, prices



    def get_class(self, mat, labels={0: -0.01, 1:0.01}):

        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)
        for k in range(len(labels)):

            if k == len(labels) - 1:
                continue

            matclasses[np.logical_and(mat > labels[k], mat <= labels[k + 1])] = k + 1
        else:

            matclasses[mat > labels[len(labels) - 1]] = len(labels)

        matclasses2 = np.zeros((matclasses.shape[0], matclasses.shape[1] - 1), dtype=np.int)

        # labels_print = range(3)
        # for i in labels_print:
        #     for j in labels_print:
        #         print(i, j, i * 3 + j)

        k = 0
        for stock in matclasses:

            two_days_classes = [(1 + len(labels)) * i + j for i, j in zip(stock[:-1], stock[1:])]
            matclasses2[k] = two_days_classes
            k += 1

        return matclasses2


    def do_train(self):
        print("Generating Train Data")
        #get data
        train_data = self.generate_train_data(column="high")
        print("Generating Train Data: Done")
        number_of_classes = 9


        # train_data = get_class(train_data, labels={0: 0})
        train_data = self.get_class(train_data)

        transision_matrix = np.zeros((self.companies_count, number_of_classes, number_of_classes))
        print("Training")
        i = 0
        for k in train_data:

            for tminus1, t in zip(k[:-1], k[1:]):
                transision_matrix[i, tminus1, t] += 1

            i += 1
        print("Training: Done")

        self.save_model(transision_matrix)
        return

    def do_eval(self):
        #### Computes overall average accuracy, per stock accuracy
        # evaluation

        transision_matrix = self.load_model()
        print("Generating Eval Data")
        eval_data, dates, prices_close = self.generate_eval_data(column="high")

        _, _, prices_low = self.generate_eval_data(column="low", stats="min")
        _, _, prices_high = self.generate_eval_data(column="high")

        _, _, prices_c = self.generate_eval_data(column="close", stats="last")
        print("Generating Eval Data: Done")
        eval_data = eval_data[:,1:]

        eval_classes = self.get_class(eval_data)

        eval_classes_new = []
        for k in eval_classes:


            lm = [k[0] // 3]
            lm += [r % 3 for r in k]

            eval_classes_new += [lm]


        def pred_all_stocks(eval_data, eval_classes):
            eval_pred = np.zeros(eval_classes.shape, dtype=np.int)
            i = 0
            for eval_element in eval_data:
                j = 0
                for state in eval_classes[i]:
                    initial_state = state
                    if j == 0:
                        eval_pred[i, j] = initial_state

                    this_trans = transision_matrix[i]
                    eval_pred[i, j] = self.predict_next_state(initial_state, this_trans)

                    j += 1
                i += 1

            return eval_pred

        eval_pred = pred_all_stocks(eval_data, eval_classes)

        eval_pred_new = []
        for k in eval_pred:
            lm = [k[0] // 3]
            lm += [r % 3 for r in k]

            eval_pred_new += [lm]
        eval_classes = np.array(eval_classes_new)
        eval_pred = np.array(eval_pred_new)


        output = dict()
        from sklearn.metrics import accuracy_score

        confidence = []

        eval_iter = 0
        company_list = self.db.get_list_companies()


        for y_true, y_pred in zip(eval_classes, eval_pred):
            comp = company_list[eval_iter]
            output[comp] = {"accuracy": accuracy_score(y_true, y_pred), "gt": y_true, "pred":y_pred,
                            "close": prices_c[eval_iter],
                            "high": prices_high[eval_iter],
                            "low": prices_low[eval_iter],
                            "dates": dates}


            confidence += [output[comp]["accuracy"]]

            eval_iter += 1

        y_true = eval_classes.flatten()
        y_pred = eval_pred.flatten()
        print(output)
        # print(y_pred)
        # for g, p, a in zip(y_true, y_pred, eval_data.flatten()):
        #     if g == p:
        #         print(g,p,a)
        print("overall accuracy: ", accuracy_score(y_true, y_pred))
        # print(confidence)
        # output["overall"] = {"accuracy": accuracy_score(y_true, y_pred)}
        # for k in output:
        #     print(k)
        #     print(k, output[k]["pred"])
        #     print(k,output[k]["gt"])
        # print(output)
        self.save_eval_output(output)


    def predict_next_state(self, initial_state, transision_matrix):

        this_trans = transision_matrix
        this_trans /= this_trans.sum(axis=1).reshape(-1, 1)
        trans_cumsum = np.cumsum(this_trans, axis=1)
        randomvalue = np.random.random()
        class_iter = 0


        for pty in trans_cumsum[initial_state]:
            if randomvalue < pty:
                break

            class_iter += 1

        return class_iter



    # def do_pred(self):
    #     #### Computes overall average accuracy, per stock accuracy
    #     # evaluation
    #     transision_matrix = self.load_model()
    #
    #     pred_data, dates, prices = self.generate_pred_data(column="close")
    #
    #     _, _, prices = self.generate_pred_data(column="open")
    #
    #     pred_classes = self.get_class(pred_data)
    #
    #     # pred_pred = np.zeros(pred_classes.shape, dtype=np.int)
    #     pred_pred = dict()
    #
    #     i = 0
    #     company_list = self.db.get_list_companies()
    #     last_date = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
    #     predict_date = last_date + datetime.timedelta(days=1)
    #     for pred_element in pred_classes:
    #         comp = company_list[i]
    #         pred_pred[comp] = {"date": str(predict_date).split(" ")[0]}
    #         pred_pred[comp]["prediction"] = self.predict_next_state(pred_element[0], transision_matrix[i])
    #
    #         i += 1
    #
    #
    #     # strategy_obj = Strategy()
    #     # resource = 500
    #     # choices, qunatities = strategy_obj.linear_problem(pred_classes[:,0], prices.T[0], resource=resource)
    #     #
    #     # print(predict_date, [company_list[k] for k in choices], qunatities, [pred_classes[k] for k in choices])
    #     # pred_pred["scenario"] = {"stocks": [company_list[k] for k in choices], "qantities": qunatities}
    #     # pred_pred["view"] = {"stocks": [company_list[k] for k in choices], "qantities": qunatities}
    #
    #
    #     self.save_pred_output(pred_pred)

    def do_pred(self, time_step = 2):
        #### Computes overall average accuracy, per stock accuracy
        # evaluation
        transision_matrix = self.load_model()
        print(transision_matrix.shape)
        import time
        start = time.time()
        pred_data, dates, prices = self.generate_pred_data(column="high")

        print(pred_data.shape)
        pred_classes = self.get_class(pred_data)
        print(pred_classes.shape)
        # _, dates, prices = self.generate_pred_data(column="close")
        end = time.time()
        print(end - start)



        # pred_pred = np.zeros(pred_classes.shape, dtype=np.int)
        pred_pred = dict()
        for t in range(time_step):
            tem_pred = []
            i = 0
            company_list = self.db.get_list_companies()

            last_date = datetime.datetime.strptime(dates[1], "%Y-%m-%d")

            predict_date = last_date + datetime.timedelta(days=7*(t+1))

            for pred_element in pred_classes:

                comp = company_list[i]
                if comp not in pred_pred.keys():
                    pred_pred[comp] = {"date": [str(predict_date).split(" ")[0]]}
                    pred_pred[comp]["prediction"] = [self.predict_next_state(pred_element[0], transision_matrix[i])]

                else:
                    pred_pred[comp]['date'] += [str(predict_date).split(" ")[0]]
                    pred_pred[comp]["prediction"] += [self.predict_next_state(pred_element[0], transision_matrix[i])]
                tem_pred += [pred_pred[comp]["prediction"]]
                i += 1
            pred_classes = tem_pred

        for each_class in pred_pred:
            test = []

            lm = [pred_pred[each_class]["prediction"][0] // 3]
            lm += [r % 3 for r in pred_pred[each_class]["prediction"]]

            test += lm
            pred_pred[each_class]["prediction"] = test[1:]

        # strategy_obj = Strategy()
        # resource = 500
        # choices, qunatities = strategy_obj.linear_problem(pred_classes[:,0], prices.T[0], resource=resource)
        #
        # print(predict_date, [company_list[k] for k in choices], qunatities, [pred_classes[k] for k in choices])
        # pred_pred["scenario"] = {"stocks": [company_list[k] for k in choices], "qantities": qunatities}
        # pred_pred["view"] = {"stocks": [company_list[k] for k in choices], "qantities": qunatities}

        print(pred_pred)
        self.save_pred_output(pred_pred)


    def save_eval_output(self, data):

        try:
            os.makedirs(self.eval_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.eval_dir)
        else:
            logging.info("Successfully created the directory %s " % self.eval_dir)

        logging.info("Writing evaluation output to %s", self.eval_file)

        with open(self.eval_file, 'wb') as outfile:
            pickle.dump(data, outfile)

        outfile.close()

    def save_pred_output(self, data):

        try:
            os.makedirs(self.pred_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.pred_dir)
        else:
            logging.info("Successfully created the directory %s " % self.pred_dir)

        logging.info("Writing evaluation output to %s", self.pred_file)

        with open(self.pred_file, 'wb') as outfile:
            pickle.dump(data, outfile)

        outfile.close()
