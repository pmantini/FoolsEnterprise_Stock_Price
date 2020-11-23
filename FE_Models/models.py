from FE_Models.model_DB_Reader import DB_Ops
import pickle, datetime, logging, os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from pyramid.arima import auto_arima

from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout, Input, Add
from keras.layers import Activation, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import Adam

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


class Markov3Class(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = ['order']
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

        self.order = None

    def do_init(self, args):

        self.order = args["order"] if "order" in args.keys() else 1
        self.name = self.name + "Order" + str(self.order)
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"

        self.model_dir = args["model_dir"] if "model_dir" in args.keys() else self.output_dir+"training_dir/"+self.name+"/"
        self.eval_dir = args["eval_dir"] if "eval_dir" in args.keys() else self.output_dir+"eval_dir/"+self.name+"/"
        self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.model_file = os.path.join(os.path.dirname(self.model_dir), "model.npy")
        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")

        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        self.companies_count = self.db.get_companies_count()

        self.n_classes = 3

    def save_model(self, transition_matrix):
        try:
            os.makedirs(self.model_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        np.save(self.model_file, transition_matrix)
        logging.info("Saving transition matrix to %s", self.model_file)

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

        train_samples_actual = values[:,:total_samples_avail]

        train_samples = (train_samples_actual[:,1:] - train_samples_actual[:,0:-1])/train_samples_actual[:,0:-1]

        #remove zeros, and infs
        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        return train_samples

    def generate_eval_data(self, column="volume"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = self.db.get_max_rows()

        values = np.zeros((total_companies, max_items))
        dates = np.array(values, dtype=object)

        i = 0
        for k in company_list:
            values_fetch, dates_fetch = self.db.get_values_company(company_sym=k, columns=column)

            dates[i, max_items - len(values_fetch):max_items] = dates_fetch
            values[i, max_items - len(values_fetch):max_items] = values_fetch
            i += 1

        total_samples_avail = max_items

        dates_samples = dates[:, -self.days_to_eval-1:total_samples_avail]

        train_samples_actual = values[:, -self.days_to_eval-self.order-1:total_samples_avail]

        train_samples = (train_samples_actual[:, 1:] - train_samples_actual[:, 0:-1]) / train_samples_actual[:, 0:-1]

        # remove zeros, and infs
        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        return train_samples, dates_samples

    def generate_pred_data(self, column="volume"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = self.db.get_max_rows()

        values = np.zeros((total_companies, max_items))
        dates = np.array(values, dtype=object)

        i = 0
        for k in company_list:
            values_fetch, dates_fetch = self.db.get_values_company(company_sym=k, columns=column)

            dates[i, max_items - len(values_fetch):max_items] = dates_fetch
            values[i, max_items - len(values_fetch):max_items] = values_fetch
            i += 1

        total_samples_avail = max_items

        dates_samples = dates[:, -1]
        train_samples_actual = values[:, -1-self.order:]

        train_samples = (train_samples_actual[:, 1:] - train_samples_actual[:, 0:-1]) / train_samples_actual[:, 0:-1]

        # remove zeros, and infs
        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        return train_samples, dates_samples, train_samples_actual[:,-1]

    def get_classes(self, data, labels={0: -0.01, 1:0.01}):
        mat = np.array(data)
        matclasses = np.zeros(mat.shape, dtype=np.int)
        classes = np.zeros((matclasses.shape[0], matclasses.shape[1] - self.order+1), dtype=np.int)
        for k in range(len(labels)):
            if k == self.n_classes - 2:
                continue
            matclasses[np.logical_and(mat > labels[k], mat <= labels[k + 1])] = k + 1
        else:
            matclasses[mat > labels[self.n_classes - 2]] = len(labels)

        class_multiplier = [k for k in reversed([(self.n_classes)**k for k in range(self.order)])]
        for ind, row in enumerate(matclasses):
            classes[ind] = [np.sum(row[k:k+self.order]*class_multiplier) for k in range(len(row)-self.order+1)]
        return classes


    def do_train(self):
        logging.info("Generating Train Data")
        train_data_changes = self.generate_train_data(column="high")

        logging.info("Generating Train Data: Done")

        classes = self.get_classes(train_data_changes)

        transition_matrix = np.zeros((self.companies_count, self.n_classes**self.order, self.n_classes**self.order))

        logging.info("Training")

        for k in range(train_data_changes.shape[0]):
            changes = np.trim_zeros(train_data_changes[k])
            tra = classes[k, -len(changes):]

            for tminus1, t in zip(tra[:-1], tra[1:]):
                transition_matrix[k, tminus1, t] += 1

        logging.info("Training: Done")

        self.save_model(transition_matrix)
        return 1


    def do_eval(self):
        logging.info("Loading transition matrix")
        transision_matrix = self.load_model()

        logging.info("Generating eval data")
        eval_data_changes, dates = self.generate_eval_data(column="high")

        classes = self.get_classes(eval_data_changes)
        eval = dict()
        class_multiplier = [k for k in reversed([(self.n_classes) ** k for k in range(self.order)])]

        for day in range(self.days_to_eval):

            logging.info("Evaluating %s" % (day))
            day_predict = []
            gt_predict = []
            dates_pred = []
            for s_ind, stock in enumerate(classes[:, day]):

                dates_pred += [dates[s_ind, day]]
                initial_state = stock
                predictions = self.predict_next_state(initial_state, transision_matrix[s_ind])

                y = predictions
                gt = classes[s_ind, day+1]
                for k in class_multiplier[:-1]:
                    gt = gt%k
                    y = y%k


                day_predict += [y]
                gt_predict += [gt]

            eval[day] = {"pred": day_predict, "gt": gt_predict, "dates":dates_pred}

        self.save_eval_output(eval)

        return 1

    def predict_next_state(self, initial_state, transition_matrix):
        this_trans = np.copy(transition_matrix)

        this_trans /= this_trans.sum(axis=1).reshape(-1, 1)
        this_trans[np.isnan(this_trans)] = 0

        trans_cumsum = np.cumsum(this_trans, axis=1)

        randomvalue = np.random.random()
        class_iter = 0

        if np.sum(trans_cumsum[initial_state]) == 0:
            logging.info("Not transision possible, inital state: %s", initial_state)
            return initial_state
        for pty in trans_cumsum[initial_state]:
            if randomvalue < pty:
                break

            class_iter += 1

        return class_iter

    def do_pred(self):
        transision_matrix = self.load_model()

        pred_data, dates, prices = self.generate_pred_data(column="high")

        pred_classes = self.get_classes(pred_data)
        # pred_pred = dict()

        tem_pred = []

        company_list = self.db.get_list_companies()
        class_multiplier = [k for k in reversed([(self.n_classes) ** k for k in range(self.order)])]
        for ind, pred_element in enumerate(pred_classes):
            prediction = self.predict_next_state(pred_element[0], transision_matrix[ind])

            for k in class_multiplier[:-1]:
                prediction = prediction % k

            tem_pred += [prediction]


        pred_pred = {"dates": dates, "predictions":tem_pred}

        self.save_pred_output(pred_pred)

        return 1

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

        logging.info("Writing prediction output to %s", self.pred_file)

        with open(self.pred_file, 'wb') as outfile:
            pickle.dump(data, outfile)

        outfile.close()


class Markov3ClassWeekly(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = ['order']
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

        self.order = None

    def do_init(self, args):

        self.order = args["order"] if "order" in args.keys() else 1
        self.name = self.name + "Order" + str(self.order)
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"

        self.model_dir = args["model_dir"] if "model_dir" in args.keys() else self.output_dir+"training_dir/"+self.name+"/"
        self.eval_dir = args["eval_dir"] if "eval_dir" in args.keys() else self.output_dir+"eval_dir/"+self.name+"/"
        self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.model_file = os.path.join(os.path.dirname(self.model_dir), "model.npy")
        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")

        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30
        self.weeks_to_eval = args["weeks_to_eval"] if "weeks_to_eval" in args.keys() else 8

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        self.companies_count = self.db.get_companies_count()

        self.n_classes = 3

    def save_model(self, transition_matrix):
        try:
            os.makedirs(self.model_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        np.save(self.model_file, transition_matrix)
        logging.info("Saving transition matrix to %s", self.model_file)

    def load_model(self):
        return np.load(self.model_file)

    def generate_train_data(self, column = "volume", stats = "max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = 0
        values_array, dates_array = [], []

        for ind, k in enumerate(company_list):
            logging.info("Fetching Data from DB: Processing - %s/%s" % (ind, len(company_list)))
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]

        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        values = np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

        total_samples_avail = max_items - self.weeks_to_eval

        train_samples = values[:, :total_samples_avail]

        train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        return train_samples

    def generate_eval_data(self, column = "volume", stats = "max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_weekly_items, max_items = 0, 0
        values_weekly_array, dates_weekly_array, values_array, dates_array = [], [], [], []

        for ind, k in enumerate(company_list):
            logging.info("Fetching Data from DB: Processing - %s/%s" % (ind, len(company_list)))
            v_temp, d_temp = self.db.get_weekly_values_company(company_sym=k, columns=column)

            values_weekly_array += [v_temp]
            dates_weekly_array += [d_temp]

            # v_daily, d_daily = self.db.get_values_company(company_sym=k, columns="close")
            #
            # values_array += [v_daily]
            # dates_array += [d_daily]

            _, d_daily = self.db.get_values_company(company_sym=k, columns="close")

            # values_array += [v_daily]
            dates_array += [d_daily]


        for k in dates_weekly_array:
            if max_weekly_items < len(k):
                max_weekly_items = len(k)

        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        # values, values_weekly = np.zeros((total_companies, max_items)), np.empty((total_companies, max_items), dtype=object)
        # dates, dates_weekly = np.array(values, dtype=object), np.array(values_weekly, dtype=object)

        values_weekly = np.empty((total_companies, max_items), dtype=object)
        dates, dates_weekly = np.array(np.zeros((total_companies, max_items)), dtype=object), np.array(values_weekly, dtype=object)

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))

            # values_fetch = values_array[i]
            # values[i, max_items - len(values_fetch):max_items] = values_fetch

            dates_fetch = dates_array[i]
            dates[i, max_items - len(dates_fetch):max_items] = dates_fetch

            values_weekly_fetch = values_weekly_array[i]
            values_weekly[i, max_weekly_items - len(values_weekly_fetch):max_weekly_items] = values_weekly_fetch

            dates_weekly_fetch = dates_weekly_array[i]

            dates_weekly[i, max_weekly_items - len(dates_weekly_fetch):max_weekly_items] = dates_weekly_fetch


        total_samples_avail = max_items
        train_samples = np.zeros(np.array(dates[:, -self.days_to_eval-1-self.order:total_samples_avail]).shape)
        dates = dates[:, -self.days_to_eval - 1:total_samples_avail]

        train_samples_weekly_prev = np.array(train_samples)
        train_samples_weekly_present = np.array(train_samples)

        for ind, comp in enumerate(dates):
            logging.info("preparing data: Processing - %s/%s" % (ind, len(company_list)))
            for day in range(len(dates[ind])):
                weeks_reveresed = [k for k in reversed(dates_weekly[ind])]
                values_reversed = [k for k in reversed(values_weekly[ind])]

                for week_ind, week in enumerate(weeks_reveresed):
                    if week is not None:
                        this_week = [datetime.datetime.strftime(k, '%Y-%m-%d') for k in week]
                        if dates[ind, day] in this_week:

                            index_day = this_week.index(dates[ind, day]) + 1

                            train_samples_weekly_prev[ind, day] = np.max(values_reversed[week_ind+1])
                            train_samples_weekly_present[ind, day] = np.max(values_reversed[week_ind][:index_day])


        train_samples = (train_samples_weekly_present - train_samples_weekly_prev)/train_samples_weekly_present
        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        return train_samples, dates


    def generate_pred_data(self, column = "volume", stats = "max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = 0
        values_array, dates_array = [], []

        for ind, k in enumerate(company_list):
            logging.info("Fetching Data from DB: Processing - %s/%s" % (ind, len(company_list)))
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]

        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        values = np.zeros((total_companies, max_items))
        dates = np.array(values, dtype=object)
        # dates_array = np.array(dates_array)
        # print(dates_array, dates_array.shape)
        # dates = dates_array[:,-self.order:]


        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            dates_fetch = dates_array[i]
            dates[i, max_items - len(dates_fetch):max_items] = dates_fetch

        total_samples_avail = max_items

        train_samples = values[:, -self.order-1:total_samples_avail]

        train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        prices = values[:, -self.order:]
        dates = dates[:, -self.order:]
        return train_samples, dates, prices

    def get_classes(self, data, labels={0: -0.01, 1:0.01}):
        mat = np.array(data)
        matclasses = np.zeros(mat.shape, dtype=np.int)
        classes = np.zeros((matclasses.shape[0], matclasses.shape[1] - self.order+1), dtype=np.int)
        for k in range(len(labels)):
            if k == self.n_classes - 2:
                continue
            matclasses[np.logical_and(mat > labels[k], mat <= labels[k + 1])] = k + 1
        else:
            matclasses[mat > labels[self.n_classes - 2]] = len(labels)

        class_multiplier = [k for k in reversed([(self.n_classes)**k for k in range(self.order)])]
        for ind, row in enumerate(matclasses):
            classes[ind] = [np.sum(row[k:k+self.order]*class_multiplier) for k in range(len(row)-self.order+1)]
        return classes


    def do_train(self):
        logging.info("Generating Train Data")
        train_data_changes = self.generate_train_data(column="high")

        logging.info("Generating Train Data: Done")

        classes = self.get_classes(train_data_changes)

        transition_matrix = np.zeros((self.companies_count, self.n_classes**self.order, self.n_classes**self.order))

        logging.info("Training")

        for k in range(train_data_changes.shape[0]):
            changes = np.trim_zeros(train_data_changes[k])
            tra = classes[k, -len(changes):]

            for tminus1, t in zip(tra[:-1], tra[1:]):
                transition_matrix[k, tminus1, t] += 1

        logging.info("Training: Done")

        self.save_model(transition_matrix)
        return

    def get_prices_eval(self, column1="high"):

        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = self.db.get_max_rows()

        values1 = np.zeros((total_companies, max_items))
        i = 0
        dates_array = np.array(values1, dtype=object)
        for k in company_list:
            values_fetch1, dates = self.db.get_values_company(company_sym=k, columns=column1)
            values1[i, max_items - len(values_fetch1):max_items] = values_fetch1

            # this_dates_array = ["-"] * max_items
            # # this_dates_array[:len(dates)] = ""
            # this_dates_array[max_items - len(dates):max_items] = dates

            dates_array[i, max_items - len(values_fetch1):max_items] = dates

            i += 1

        train_samples1 = values1[:, (-self.weeks_to_eval-1)*7:]
        dates_samples = dates_array[:,(-self.weeks_to_eval-1)*7:]

        return train_samples1, dates_samples

    def do_eval(self):
        logging.info("Loading transition matrix")
        transision_matrix = self.load_model()

        logging.info("Generating eval data")
        eval_data_changes, dates = self.generate_eval_data(column="high")

        classes = self.get_classes(eval_data_changes)
        eval = dict()
        class_multiplier = [k for k in reversed([(self.n_classes) ** k for k in range(self.order)])]

        for day in range(self.days_to_eval):

            logging.info("Evaluating %s" % (day))
            day_predict = []
            gt_predict = []
            dates_pred = []
            for s_ind, stock in enumerate(classes[:, day]):

                dates_pred += [dates[s_ind, day]]
                initial_state = stock
                predictions = self.predict_next_state(initial_state, transision_matrix[s_ind])

                y = predictions
                gt = classes[s_ind, day + 1]
                for k in class_multiplier[:-1]:
                    gt = gt % k
                    y = y % k

                day_predict += [y]
                gt_predict += [gt]

            eval[day] = {"pred": day_predict, "gt": gt_predict, "dates": dates_pred}

        self.save_eval_output(eval)

        return 1

    def predict_next_state(self, initial_state, transition_matrix):
        this_trans = np.copy(transition_matrix)

        this_trans /= this_trans.sum(axis=1).reshape(-1, 1)
        this_trans[np.isnan(this_trans)] = 0

        trans_cumsum = np.cumsum(this_trans, axis=1)

        randomvalue = np.random.random()
        class_iter = 0

        if np.sum(trans_cumsum[initial_state]) == 0:
            logging.info("Not transision possible, inital state: %s", initial_state)
            return initial_state
        for pty in trans_cumsum[initial_state]:
            if randomvalue < pty:
                break

            class_iter += 1

        return class_iter

    # def do_pred(self):
    #     transision_matrix = self.load_model()
    #
    #     # import time
    #     # start = time.time()
    #     pred_data, dates, prices = self.generate_pred_data(column="high")
    #
    #     pred_classes = self.get_classes(pred_data)
    #
    #     # end = time.time()
    #     # logging.info(end - start)
    #
    #     pred_pred = dict()
    #     # for t in range(time_step):
    #     tem_pred = []
    #
    #     company_list = self.db.get_list_companies()
    #
    #     for ind, pred_element in enumerate(pred_classes):
    #
    #         last_date = datetime.datetime.strptime(dates[ind, -1], "%Y-%m-%d")
    #
    #         predict_date = last_date + datetime.timedelta(days=7)
    #
    #         comp = company_list[ind]
    #
    #         pred_pred[comp] = {"date": [str(predict_date).split(" ")[0]]}
    #         pred_pred[comp]["prediction"] = [self.predict_next_state(pred_element[0], transision_matrix[ind])]
    #
    #     class_multiplier = [k for k in reversed([(self.n_classes) ** k for k in range(self.order)])]
    #     for each_class in pred_pred:
    #
    #         this_pred = pred_pred[each_class]["prediction"]
    #         for k in class_multiplier[:-1]:
    #             this_pred = [r % k for r in this_pred]
    #
    #         pred_pred[each_class]["prediction"] = this_pred
    #
    #     self.save_pred_output(pred_pred)
    #
    #     return 1

    def do_pred(self):
        transision_matrix = self.load_model()

        pred_data, dates, prices = self.generate_pred_data(column="high")

        pred_classes = self.get_classes(pred_data)
        # pred_pred = dict()

        tem_pred = []

        company_list = self.db.get_list_companies()
        class_multiplier = [k for k in reversed([(self.n_classes) ** k for k in range(self.order)])]
        for ind, pred_element in enumerate(pred_classes):
            prediction = self.predict_next_state(pred_element[0], transision_matrix[ind])

            for k in class_multiplier[:-1]:
                prediction = prediction % k

            tem_pred += [prediction]


        pred_pred = {"dates": dates, "predictions":tem_pred}

        self.save_pred_output(pred_pred)

        return 1

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

        logging.info("Writing prediction output to %s", self.pred_file)

        with open(self.pred_file, 'wb') as outfile:
            pickle.dump(data, outfile)

        outfile.close()
