from keras.layers import Dense, Flatten, Dropout, BatchNormalization, LSTM
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

import numpy as np

import logging
import os
import shutil



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
        for k in company_list:
            _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
            if max_items < len(dates_fetch):
                max_items = len(dates_fetch)

        i = 0
        for k in company_list:
            if i == 0:
                _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
                #max_items = len(dates_fetch)
                values = np.zeros((total_companies, max_items))
            values_fetch, _ = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)

            values[i, max_items - len(values_fetch):max_items] = values_fetch
            i += 1


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
        for k in company_list:
            _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
            if max_items < len(dates_fetch) - 1:
                max_items = len(dates_fetch) - 1


        i = 0
        for k in company_list:

            if i == 0:
                _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k, stats=stats)
                # max_items = len(dates_fetch) - 1

                values = np.zeros((total_companies, max_items))

            values_fetch, _ = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_fetch = values_fetch[:-from_end]


            values[i, max_items - len(values_fetch):max_items] = values_fetch
            i += 1


        dates = dates_fetch[-weeks_to_eval-2:]

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

        max_items = 0
        for k in company_list:
            _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k)
            if max_items < len(dates_fetch) - 1:
                max_items = len(dates_fetch) - 1

        i = 0
        for k in company_list:
            if i == 0:
                _, dates_fetch = self.db.get_weekly_stats_company(company_sym=k)
                # max_items = len(dates_fetch) - 1

                values = np.zeros((total_companies, max_items))

            values_fetch, _ = self.db.get_weekly_stats_company(company_sym=k, columns=column)
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

        pred_data, dates, prices = self.generate_pred_data(column="high")
        _, dates, prices = self.generate_pred_data(column="close")

        pred_classes = self.get_class(pred_data)


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



