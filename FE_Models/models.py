from FE_Models.model_DB_Reader import DB_Ops
import pickle, datetime, logging, os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from pyramid.arima import auto_arima

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

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


class Cnn3ClassWeekly(FEModel):

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

        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30
        self.weeks_to_eval = args["weeks_to_eval"] if "weeks_to_eval" in args.keys() else 4

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()
        self.company_list = self.db.get_list_companies()
        self.companies_count = self.db.get_companies_count()

        self.n_steps = 10
        self.n_classes = 3

        self.model = self.init_model()
        

    def init_model(self):

        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.n_steps, 1)))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def save_model(self, model, stock):
        file_name = stock+"_arima.pkl"
        try:
            os.makedirs(self.model_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        with open(os.path.join(self.model_dir, file_name), 'wb') as pkl:
            pickle.dump(model, pkl)

        logging.info("Saving transition matrix to %s", self.model_file)

    def load_model(self, stock):
        file_name = stock + "_arima.pkl"
        with open(file_name, 'rb') as pkl:
            model = pickle.load(pkl)
        return model

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

        total_samples_avail = max_items

        train_samples = values[:, :total_samples_avail]


        train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]


        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0


        return train_samples

    def get_class_1stOrder(self, mat, labels={0: -0.01, 1:0.01}):

        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)
        for k in range(len(labels)):
            if k == len(labels) - 1:
                continue
            matclasses[np.logical_and(mat > labels[k], mat <= labels[k + 1])] = k + 1
        else:
            matclasses[mat > labels[len(labels) - 1]] = len(labels)

        return matclasses


    def get_train_test(self, rows):
        x_train = []
        y_train = []
        for k in range(len(rows)-self.n_steps-1):
            x_train += [rows[k:k+self.n_steps]]
            y_train += [rows[k+self.n_steps]]

        return np.expand_dims(np.array(x_train), axis=2), y_train

    def do_train(self):
        logging.info("Generating Train Data")
        train_data_changes = self.generate_train_data(column="high")

        logging.info("Generating Train Data: Done")

        classes_train = self.get_class_1stOrder(train_data_changes)

        logging.info("Training")

        for ind, k in enumerate(classes_train):

            trimmed = np.trim_zeros(train_data_changes[ind])

            x_train, y_train  = self.get_train_test(k[-len(trimmed):])
            y_train = to_categorical(y_train, num_classes=self.n_classes)

            self.model.fit(x_train, y_train, epochs=100)


            #
            # changes = np.trim_zeros(train_data_changes[k])
            # # tra_2 = order_1_classes[k, -len(changes):]
            # model = auto_arima(changes, trace=False, error_action='ignore', suppress_warnings=True)
            # model.fit(changes)
            #
            # self.save_model(model, self.company_list[k])

        logging.info("Training: Done")

        # self.save_model(transition_matrix)
        return

    def do_eval(self):

        logging.info("Generating eval data")
        eval_data_changes, eval_data_changes_a = self.generate_train_data(column="high")
        logging.info("Generating eval data: Done")

        order_1_classes = self.get_class_1stOrder(eval_data_changes)

        logging.info("Evaluating prediction accuracy")
        prediction_accuracies = []

        y_true_overall = []
        y_pred_overall = []

        for ind, row in enumerate(eval_data_changes):
            row = np.trim_zeros(row)

            length_data = len(row)
            pred = []
            for k in range(self.weeks_to_eval):
                train_data = row[:(length_data-self.weeks_to_eval+k)]
                model = auto_arima(train_data, error_action='ignore', suppress_warnings=True, seasonal=True)
                model.fit(train_data)
                pred += [model.predict(n_periods=1)[0]]

            y_true = order_1_classes[ind][-self.weeks_to_eval:]
            y_pred = self.get_class_1stOrder(pred)

            prediction_accuracies += [accuracy_score(y_true, y_pred)]
            logging.info("%s, 1st Order, GT: %s, Pred: %s, acc: %s" % (ind, y_true, y_pred, prediction_accuracies[-1]))

            y_true_overall += [y_true]
            y_pred_overall += [y_pred]

        y_true_overall = np.array(y_true_overall).flatten()
        y_pred_overall = np.array(y_pred_overall).flatten()
        logging.info("Overall accuracy, 1st order: %s" % (np.mean(prediction_accuracies)))

        confusion = np.array(confusion_matrix(y_true_overall, y_pred_overall)).astype("float")
        total = np.sum(confusion, axis=1)
        confusion = np.array([r / l for r, l in zip(confusion, total)])
        logging.info("Confusion matrix \n%s" % confusion)
        logging.info("Risk: %s" % (confusion[0, 2]))

    def predict_next_state(self, initial_state, transition_matrix):
        this_trans = np.copy(transition_matrix)

        this_trans /= this_trans.sum(axis=1).reshape(-1, 1)
        this_trans[np.isnan(this_trans)] = 0

        trans_cumsum = np.cumsum(this_trans, axis=1)

        randomvalue = np.random.random()
        class_iter = 0

        if np.sum(trans_cumsum[initial_state]) == 0:
            print("Not transision possible, inital state: ", initial_state)
            return initial_state
        for pty in trans_cumsum[initial_state]:
            if randomvalue < pty:
                break

            class_iter += 1

        return class_iter

    def do_pred(self, time_step = 2):
        transision_matrix = self.load_model()

        import time
        start = time.time()
        pred_data, dates, prices = self.generate_pred_data(column="high")

        pred_classes = self.get_class(pred_data)

        end = time.time()
        logging.info(end - start)

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


class Arima3ClassWeekly(FEModel):

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

        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30
        self.weeks_to_eval = args["weeks_to_eval"] if "weeks_to_eval" in args.keys() else 4

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()
        self.company_list = self.db.get_list_companies()
        self.companies_count = self.db.get_companies_count()

    def save_model(self, model, stock):
        file_name = stock+"_arima.pkl"
        try:
            os.makedirs(self.model_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        with open(os.path.join(self.model_dir, file_name), 'wb') as pkl:
            pickle.dump(model, pkl)

        logging.info("Saving transition matrix to %s", self.model_file)

    def load_model(self, stock):
        file_name = stock + "_arima.pkl"
        with open(file_name, 'rb') as pkl:
            model = pickle.load(pkl)
        return model

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

        total_samples_avail = max_items

        train_samples = values[:, :total_samples_avail]

        train_samples_a = (train_samples[:, 1:] - train_samples[:, 0:-1])
        train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]


        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        # train_samples_a[np.isnan(train_samples_a)] = 0
        # train_samples_a[np.isinf(train_samples_a)] = 0

        return train_samples, train_samples_a

    def get_class_1stOrder(self, mat, labels={0: -0.01, 1:0.01}):

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
        logging.info("Generating Train Data")
        train_data_changes = self.generate_train_data(column="high")

        logging.info("Generating Train Data: Done")

        logging.info("Training")

        for k in range(train_data_changes.shape[0]):
            changes = np.trim_zeros(train_data_changes[k])
            # tra_2 = order_1_classes[k, -len(changes):]
            model = auto_arima(changes, trace=False, error_action='ignore', suppress_warnings=True)
            model.fit(changes)

            self.save_model(model, self.company_list[k])

        logging.info("Training: Done")

        # self.save_model(transition_matrix)
        return

    def do_eval(self):

        logging.info("Generating eval data")
        eval_data_changes, eval_data_changes_a = self.generate_train_data(column="high")
        logging.info("Generating eval data: Done")

        order_1_classes = self.get_class_1stOrder(eval_data_changes)

        logging.info("Evaluating prediction accuracy")
        prediction_accuracies = []

        y_true_overall = []
        y_pred_overall = []

        for ind, row in enumerate(eval_data_changes):
            row = np.trim_zeros(row)

            length_data = len(row)
            pred = []
            for k in range(self.weeks_to_eval):
                train_data = row[:(length_data-self.weeks_to_eval+k)]
                model = auto_arima(train_data, error_action='ignore', suppress_warnings=True, seasonal=True)
                model.fit(train_data)
                pred += [model.predict(n_periods=1)[0]]

            y_true = order_1_classes[ind][-self.weeks_to_eval:]
            y_pred = self.get_class_1stOrder(pred)

            prediction_accuracies += [accuracy_score(y_true, y_pred)]
            logging.info("%s, 1st Order, GT: %s, Pred: %s, acc: %s" % (ind, y_true, y_pred, prediction_accuracies[-1]))

            y_true_overall += [y_true]
            y_pred_overall += [y_pred]

        y_true_overall = np.array(y_true_overall).flatten()
        y_pred_overall = np.array(y_pred_overall).flatten()
        logging.info("Overall accuracy, 1st order: %s" % (np.mean(prediction_accuracies)))

        confusion = np.array(confusion_matrix(y_true_overall, y_pred_overall)).astype("float")
        total = np.sum(confusion, axis=1)
        confusion = np.array([r / l for r, l in zip(confusion, total)])
        logging.info("Confusion matrix \n%s" % confusion)
        logging.info("Risk: %s" % (confusion[0, 2]))

    def predict_next_state(self, initial_state, transition_matrix):
        this_trans = np.copy(transition_matrix)

        this_trans /= this_trans.sum(axis=1).reshape(-1, 1)
        this_trans[np.isnan(this_trans)] = 0

        trans_cumsum = np.cumsum(this_trans, axis=1)

        randomvalue = np.random.random()
        class_iter = 0

        if np.sum(trans_cumsum[initial_state]) == 0:
            print("Not transision possible, inital state: ", initial_state)
            return initial_state
        for pty in trans_cumsum[initial_state]:
            if randomvalue < pty:
                break

            class_iter += 1

        return class_iter

    def do_pred(self, time_step = 2):
        transision_matrix = self.load_model()

        import time
        start = time.time()
        pred_data, dates, prices = self.generate_pred_data(column="high")

        pred_classes = self.get_class(pred_data)

        end = time.time()
        logging.info(end - start)

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

class Markov1stOrder3ClassWeeklyCRFVolume(FEModel):

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

        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30
        self.weeks_to_eval = args["weeks_to_eval"] if "weeks_to_eval" in args.keys() else 4

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        self.companies_count = self.db.get_companies_count()

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
        values_array, dates_array, volume_array = [], [], []

        for ind, k in enumerate(company_list):
            logging.info("Fetching Data from DB: Processing - %s/%s" % (ind, len(company_list)))
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]
            vol_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="volume", stats=stats)
            volume_array += [vol_temp]

        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        values = np.zeros((total_companies, max_items))
        volume_values= np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

        total_samples_avail = max_items

        train_samples = values[:, :total_samples_avail]
        train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample =  volume_values[:, :total_samples_avail]
        volume_sample = (volume_sample[:,1:] - volume_sample[:,0:-1])/volume_sample[:,0:-1]

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0


        return train_samples, volume_sample

    def get_class_1stOrder(self, mat, labels={0: -0.01, 1:0.01}):

        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)
        for k in range(len(labels)):
            if k == len(labels) - 1:
                continue
            matclasses[np.logical_and(mat > labels[k], mat <= labels[k + 1])] = k + 1
        else:
            matclasses[mat > labels[len(labels) - 1]] = len(labels)

        return matclasses

    def get_class_1stOrder_mean(self, mat):

        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)
        for k in range(mat.shape[0]):
            temp = np.zeros(mat[k].shape)
            temp[mat[k] > np.mean(mat[k])] = 1
            matclasses[k] = temp

        return matclasses

    def get_conditional_classes(self, classes_price, classes_volume):
        return (np.max(classes_price)+1)*classes_volume + classes_price

    def do_train(self):
        logging.info("Generating Train Data")
        train_data_changes, volume_change = self.generate_train_data(column="high")
        logging.info("Generating Train Data: Done")

        number_of_classes = 6
        order_1_classes = self.get_class_1stOrder(train_data_changes)
        order_1_classes_volume = self.get_class_1stOrder_mean(volume_change)


        conditional_classes = self.get_conditional_classes(order_1_classes, order_1_classes_volume)

        transition_matrix = np.zeros((self.companies_count, number_of_classes, number_of_classes))

        logging.info("Training")

        for k in range(train_data_changes.shape[0]):
            changes = np.trim_zeros(train_data_changes[k])
            tra_2 = conditional_classes[k, -len(changes):]
            for tminus1, t in zip(tra_2[:-1], tra_2[1:]):
                transition_matrix[k, tminus1, t] += 1

        logging.info("Training: Done")

        self.save_model(transition_matrix)
        return

    def do_eval(self):
        logging.info("Loading transition matrix")
        transision_matrix = self.load_model()

        logging.info("Generating eval data: prices")
        eval_price_changes, eval_volume_change = self.generate_train_data(column="high")
        logging.info("Generating eval data: prices - Done")

        order_1_classes = self.get_class_1stOrder(eval_price_changes)
        order_1_classes_volume = self.get_class_1stOrder_mean(eval_volume_change)

        conditional_classes = self.get_conditional_classes(order_1_classes, order_1_classes_volume)

        logging.info("Evaluating prediction accuracy")
        prediction_accuracies = []

        y_true_overall = []
        y_pred_overall = []

        for ind, row in enumerate(conditional_classes):
            eval_data_for_this_row = row[-self.weeks_to_eval-1:]
            y_true = eval_data_for_this_row[1:]
            y_pred = []
            for k in eval_data_for_this_row[:-1]:
                initial_state = k
                y_pred += [self.predict_next_state(initial_state, transision_matrix[ind])]

            y_true_order_1 = [r % 3 for r in y_true]
            y_pred_order_1 = [r % 3 for r in y_pred]


            prediction_accuracies += [accuracy_score(y_true_order_1, y_pred_order_1)]
            logging.info("%s, 1st Order, GT: %s, Pred: %s, acc: %s" % (ind, y_true_order_1, y_pred_order_1, prediction_accuracies[-1]))

            y_true_overall += [y_true_order_1]
            y_pred_overall += [y_pred_order_1]

        y_true_overall = np.array(y_true_overall).flatten()
        y_pred_overall = np.array(y_pred_overall).flatten()
        logging.info("Overall accuracy, 1st order: %s" % (np.mean(prediction_accuracies)))

        confusion = np.array(confusion_matrix(y_true_overall, y_pred_overall)).astype("float")
        total = np.sum(confusion, axis=1)
        confusion = np.array([r/l for r, l in zip(confusion,total)])
        logging.info("Confusion matrix \n%s" % confusion)
        logging.info("Risk: %s" % (confusion[0, 2]))

    def predict_next_state(self, initial_state, transition_matrix):
        this_trans = np.copy(transition_matrix)

        this_trans /= this_trans.sum(axis=1).reshape(-1, 1)
        this_trans[np.isnan(this_trans)] = 0

        trans_cumsum = np.cumsum(this_trans, axis=1)

        randomvalue = np.random.random()
        class_iter = 0

        if np.sum(trans_cumsum[initial_state]) == 0:
            print("Not transision possible, inital state: ", initial_state)
            return initial_state
        for pty in trans_cumsum[initial_state]:
            if randomvalue < pty:
                break

            class_iter += 1

        return class_iter

    def do_pred(self, time_step = 2):
        transision_matrix = self.load_model()

        import time
        start = time.time()
        pred_data, dates, prices = self.generate_pred_data(column="high")

        pred_classes = self.get_class(pred_data)

        end = time.time()
        logging.info(end - start)

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


class Markov1stOrder3ClassWeekly(FEModel):

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

        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30
        self.weeks_to_eval = args["weeks_to_eval"] if "weeks_to_eval" in args.keys() else 4

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        self.companies_count = self.db.get_companies_count()

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

        total_samples_avail = max_items

        train_samples = values[:, :total_samples_avail]

        train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        return train_samples

    def get_class_1stOrder(self, mat, labels={0: -0.01, 1:0.01}):

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
        logging.info("Generating Train Data")
        train_data_changes = self.generate_train_data(column="high")

        logging.info("Generating Train Data: Done")
        number_of_classes = 3
        order_1_classes = self.get_class_1stOrder(train_data_changes)

        transition_matrix = np.zeros((self.companies_count, number_of_classes, number_of_classes))

        logging.info("Training")

        for k in range(train_data_changes.shape[0]):
            changes = np.trim_zeros(train_data_changes[k])
            tra_2 = order_1_classes[k, -len(changes):]
            for tminus1, t in zip(tra_2[:-1], tra_2[1:]):
                transition_matrix[k, tminus1, t] += 1

        logging.info("Training: Done")

        self.save_model(transition_matrix)
        return

    def do_eval(self):
        logging.info("Loading transition matrix")
        transision_matrix = self.load_model()

        logging.info("Generating eval data")
        eval_data_changes = self.generate_train_data(column="high")
        logging.info("Generating eval data: Done")

        order_1_classes = self.get_class_1stOrder(eval_data_changes)

        logging.info("Evaluating prediction accuracy")
        prediction_accuracies = []

        y_true_overall = []
        y_pred_overall = []

        for ind, row in enumerate(order_1_classes):
            eval_data_for_this_row = row[-self.weeks_to_eval-1:]
            y_true = eval_data_for_this_row[1:]
            y_pred = []
            for k in eval_data_for_this_row[:-1]:
                initial_state = k
                y_pred += [self.predict_next_state(initial_state, transision_matrix[ind])]

            prediction_accuracies += [accuracy_score(y_true, y_pred)]
            logging.info("%s, 1st Order, GT: %s, Pred: %s, acc: %s" % (ind, y_true, y_pred, prediction_accuracies[-1]))

            y_true_overall += [y_true]
            y_pred_overall += [y_pred]

        y_true_overall = np.array(y_true_overall).flatten()
        y_pred_overall = np.array(y_pred_overall).flatten()
        logging.info("Overall accuracy, 1st order: %s" % (np.mean(prediction_accuracies)))

        confusion = np.array(confusion_matrix(y_true_overall, y_pred_overall)).astype("float")
        total = np.sum(confusion, axis=1)
        confusion = np.array([r/l for r, l in zip(confusion,total)])
        logging.info("Confusion matrix \n%s" % confusion)
        logging.info("Risk: %s" % (confusion[0,2]))

    def predict_next_state(self, initial_state, transition_matrix):
        this_trans = np.copy(transition_matrix)

        this_trans /= this_trans.sum(axis=1).reshape(-1, 1)
        this_trans[np.isnan(this_trans)] = 0

        trans_cumsum = np.cumsum(this_trans, axis=1)

        randomvalue = np.random.random()
        class_iter = 0

        if np.sum(trans_cumsum[initial_state]) == 0:
            print("Not transision possible, inital state: ", initial_state)
            return initial_state
        for pty in trans_cumsum[initial_state]:
            if randomvalue < pty:
                break

            class_iter += 1

        return class_iter

    def do_pred(self, time_step = 2):
        transision_matrix = self.load_model()

        import time
        start = time.time()
        pred_data, dates, prices = self.generate_pred_data(column="high")

        pred_classes = self.get_class(pred_data)

        end = time.time()
        logging.info(end - start)

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



class Markov2ndOrder3ClassWeekly(FEModel):

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

        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30
        self.weeks_to_eval = args["weeks_to_eval"] if "weeks_to_eval" in args.keys() else 4

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        self.companies_count = self.db.get_companies_count()

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

        total_samples_avail = max_items

        train_samples = values[:, :total_samples_avail]

        train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        return train_samples

    def get_class_1stOrder(self, mat, labels={0: -0.01, 1:0.01}):

        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)
        for k in range(len(labels)):
            if k == len(labels) - 1:
                continue
            matclasses[np.logical_and(mat > labels[k], mat <= labels[k + 1])] = k + 1
        else:
            matclasses[mat > labels[len(labels) - 1]] = len(labels)

        return matclasses

    def get_class_2ndOrder(self, matclasses, labels={0: -0.01, 1:0.01}):
        matclasses2 = np.zeros((matclasses.shape[0], matclasses.shape[1] - 1), dtype=np.int)

        k = 0
        for stock in matclasses:
            two_days_classes = [(1 + len(labels)) * i + j for i, j in zip(stock[:-1], stock[1:])]
            matclasses2[k] = two_days_classes
            k += 1
        return matclasses2

    def do_train(self):
        logging.info("Generating Train Data")
        train_data_changes = self.generate_train_data(column="high")

        logging.info("Generating Train Data: Done")
        number_of_classes = 9
        order_1_classes = self.get_class_1stOrder(train_data_changes)
        order_2_classes = self.get_class_2ndOrder(order_1_classes)

        transition_matrix = np.zeros((self.companies_count, number_of_classes, number_of_classes))

        logging.info("Training")

        for k in range(train_data_changes.shape[0]):
            changes = np.trim_zeros(train_data_changes[k])
            tra_2 = order_2_classes[k, -len(changes):]
            for tminus1, t in zip(tra_2[:-1], tra_2[1:]):
                transition_matrix[k, tminus1, t] += 1

        logging.info("Training: Done")

        self.save_model(transition_matrix)
        return

    def do_eval(self):
        logging.info("Loading transition matrix")
        transision_matrix = self.load_model()

        logging.info("Generating eval data")
        eval_data_changes = self.generate_train_data(column="high")
        logging.info("Generating eval data: Done")

        order_1_classes = self.get_class_1stOrder(eval_data_changes)
        order_2_classes = self.get_class_2ndOrder(order_1_classes)

        logging.info("Evaluating prediction accuracy")
        prediction_accuracies_1, prediction_accuracies_2 = [], []

        y_true_overall = []
        y_pred_overall = []

        y_true_overall_2 = []
        y_pred_overall_2 = []

        for ind, row in enumerate(order_2_classes):
            eval_data_for_this_row = row[-self.weeks_to_eval-1:]
            y_true = eval_data_for_this_row[1:]
            y_pred = []
            for k in eval_data_for_this_row[:-1]:
                initial_state = k
                y_pred += [self.predict_next_state(initial_state, transision_matrix[ind])]
            # prediction_accuracies_2 += [accuracy_score(y_true, y_pred)]
            # logging.info("%s, 2nd Order, GT: %s, Pred: %s, acc: %s" % (ind, y_true, y_pred, prediction_accuracies_2[-1]))

            y_true_overall_2 += [r for r in y_true]
            y_pred_overall_2 += [r for r in y_pred]

            prediction_accuracies_2 += [accuracy_score(y_true, y_pred)]

            y_true_order_1 = [r%3 for r in y_true]
            y_pred_order_1 = [r%3 for r in y_pred]
            prediction_accuracies_1 += [accuracy_score(y_true_order_1, y_pred_order_1)]
            logging.info("%s, 1st Order, GT: %s, Pred: %s, acc: %s" % (ind, y_true_order_1, y_pred_order_1, prediction_accuracies_1[-1]))

            y_true_overall += y_true_order_1
            y_pred_overall += y_pred_order_1


        logging.info("Overall accuracy, 1st order: %s" % (np.mean(prediction_accuracies_1)))

        confusion = np.array(confusion_matrix(y_true_overall, y_pred_overall)).astype("float")
        total = np.sum(confusion, axis=1)
        confusion = np.array([r/l for r, l in zip(confusion,total)])
        logging.info("Confusion matrix \n%s" % confusion)
        logging.info("Risk: %s" % (confusion[0,2]))

        logging.info("Overall accuracy, 2nd order: %s" % (np.mean(prediction_accuracies_2)))

        confusion = np.array(confusion_matrix(y_true_overall_2, y_pred_overall_2)).astype("float")
        total = np.sum(confusion, axis=1)
        confusion = np.array([r / l for r, l in zip(confusion, total)])
        logging.info("Confusion matrix \n%s" % confusion)


        # logging.info("Risk: %s" % (confusion[0, 2]))



    def predict_next_state(self, initial_state, transition_matrix):
        this_trans = np.copy(transition_matrix)

        this_trans /= this_trans.sum(axis=1).reshape(-1, 1)
        this_trans[np.isnan(this_trans)] = 0

        trans_cumsum = np.cumsum(this_trans, axis=1)

        randomvalue = np.random.random()
        class_iter = 0

        if np.sum(trans_cumsum[initial_state]) == 0:
            print("Not transision possible, inital state: ", initial_state)
            return initial_state
        for pty in trans_cumsum[initial_state]:
            if randomvalue < pty:
                break

            class_iter += 1

        return class_iter

    def do_pred(self, time_step = 2):
        transision_matrix = self.load_model()

        import time
        start = time.time()
        pred_data, dates, prices = self.generate_pred_data(column="high")

        pred_classes = self.get_class(pred_data)

        end = time.time()
        logging.info(end - start)

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

class Markov3rdOrder3ClassWeekly(FEModel):

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

        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30
        self.weeks_to_eval = args["weeks_to_eval"] if "weeks_to_eval" in args.keys() else 4

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        self.companies_count = self.db.get_companies_count()

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

        total_samples_avail = max_items

        train_samples = values[:, :total_samples_avail]

        train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        return train_samples

    def get_class_1stOrder(self, mat, labels={0: -0.01, 1:0.01}):

        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)
        for k in range(len(labels)):
            if k == len(labels) - 1:
                continue
            matclasses[np.logical_and(mat > labels[k], mat <= labels[k + 1])] = k + 1
        else:
            matclasses[mat > labels[len(labels) - 1]] = len(labels)

        return matclasses

    def get_class_2ndOrder(self, matclasses, labels={0: -0.01, 1:0.01}):
        matclasses2 = np.zeros((matclasses.shape[0], matclasses.shape[1] - 1), dtype=np.int)

        k = 0
        for stock in matclasses:
            two_days_classes = [(1 + len(labels)) * i + j for i, j in zip(stock[:-1], stock[1:])]
            matclasses2[k] = two_days_classes
            k += 1
        return matclasses2

    def get_class_3rdOrder(self, matclasses, matclasses2, labels={0: -0.01, 1:0.01}):
        matclasses3 = np.zeros((matclasses2.shape[0], matclasses2.shape[1] - 1), dtype=np.int)

        k = 0
        for first, second in zip(matclasses, matclasses2):
            matclasses3[k] = 9*first[:-2]+second[1:]
            k += 1

        return matclasses3

    def do_train(self):
        logging.info("Generating Train Data")
        train_data_changes = self.generate_train_data(column="high")

        logging.info("Generating Train Data: Done")
        number_of_classes = 27
        order_1_classes = self.get_class_1stOrder(train_data_changes)
        order_2_classes = self.get_class_2ndOrder(order_1_classes)
        order_3_classes = self.get_class_3rdOrder(order_1_classes, order_2_classes)

        transition_matrix = np.zeros((self.companies_count, number_of_classes, number_of_classes))

        logging.info("Training")

        for k in range(train_data_changes.shape[0]):
            changes = np.trim_zeros(train_data_changes[k])
            tra_3 = order_3_classes[k, -len(changes):]
            for tminus1, t in zip(tra_3[:-1], tra_3[1:]):
                transition_matrix[k, tminus1, t] += 1

        logging.info("Training: Done")

        self.save_model(transition_matrix)
        return

    def do_eval(self):
        logging.info("Loading transition matrix")
        transision_matrix = self.load_model()

        logging.info("Generating eval data")
        eval_data_changes = self.generate_train_data(column="high")
        logging.info("Generating eval data: Done")

        order_1_classes = self.get_class_1stOrder(eval_data_changes)
        order_2_classes = self.get_class_2ndOrder(order_1_classes)
        order_3_classes = self.get_class_3rdOrder(order_1_classes, order_2_classes)

        logging.info("Evaluating prediction accuracy")

        y_true_overall, y_true_overall_3 = [], []
        y_pred_overall, y_pred_overall_3 = [], []
        prediction_accuracies_1, prediction_accuracies_3 = [], []
        for ind, row in enumerate(order_3_classes):

            eval_data_for_this_row = row[-self.weeks_to_eval-1:]

            y_true = eval_data_for_this_row[1:]
            y_pred = []
            for k in eval_data_for_this_row[:-1]:

                initial_state = k
                y_pred += [self.predict_next_state(initial_state, transision_matrix[ind])]

            y_true_overall_3 += [r for r in y_true]
            y_pred_overall_3 += [r for r in y_pred]

            prediction_accuracies_3 += [accuracy_score(y_true, y_pred)]

            y_true_order_1 = [r%9%3 for r in y_true]
            y_pred_order_1 = [r%9%3 for r in y_pred]
            prediction_accuracies_1 += [accuracy_score(y_true_order_1, y_pred_order_1)]
            logging.info("%s, 1st Order, GT: %s, Pred: %s, acc: %s" % (ind, y_true_order_1, y_pred_order_1, prediction_accuracies_1[-1]))

            y_true_overall += y_true_order_1
            y_pred_overall += y_pred_order_1

        logging.info("Overall accuracy, 1st order: %s" % (np.mean(prediction_accuracies_1)))

        confusion = np.array(confusion_matrix(y_true_overall, y_pred_overall)).astype("float")
        total = np.sum(confusion, axis=1)
        confusion = np.array([r/l for r, l in zip(confusion,total)])
        logging.info("Confusion matrix \n%s" % confusion)
        logging.info("Risk: %s" % (confusion[0,2]))

        logging.info("Overall accuracy, 3rd order: %s" % (np.mean(prediction_accuracies_3)))

        confusion = np.array(confusion_matrix(y_true_overall_3, y_pred_overall_3)).astype("float")
        total = np.sum(confusion, axis=1)
        confusion = np.array([r / l for r, l in zip(confusion, total)])
        # logging.info("Confusion matrix \n%s" % confusion)
        # for i in range(0, 27, 3):
        #     print(i, confusion[i:i+3,i:i+3])
        #
        # plt.matshow(confusion)
        # plt.show()

        # y_true_overall = []
        # y_pred_overall = []
        #
        # y_true_overall_2 = []
        # y_pred_overall_2 = []
        #
        # for ind, row in enumerate(order_2_classes):
        #     eval_data_for_this_row = row[-self.weeks_to_eval-1:]
        #     y_true = eval_data_for_this_row[1:]
        #     y_pred = []
        #     for k in eval_data_for_this_row[:-1]:
        #         initial_state = k
        #         y_pred += [self.predict_next_state(initial_state, transision_matrix[ind])]
        #     # prediction_accuracies_2 += [accuracy_score(y_true, y_pred)]
        #     # logging.info("%s, 2nd Order, GT: %s, Pred: %s, acc: %s" % (ind, y_true, y_pred, prediction_accuracies_2[-1]))
        #
        #     y_true_overall_2 += [r for r in y_true]
        #     y_pred_overall_2 += [r for r in y_pred]
        #
        #     prediction_accuracies_2 += [accuracy_score(y_true, y_pred)]
        #
        #     y_true_order_1 = [r%3 for r in y_true]
        #     y_pred_order_1 = [r%3 for r in y_pred]
        #     prediction_accuracies_1 += [accuracy_score(y_true_order_1, y_pred_order_1)]
        #     logging.info("%s, 1st Order, GT: %s, Pred: %s, acc: %s" % (ind, y_true_order_1, y_pred_order_1, prediction_accuracies_1[-1]))
        #
        #     y_true_overall += y_true_order_1
        #     y_pred_overall += y_pred_order_1


        # logging.info("Overall accuracy, 1st order: %s" % (np.mean(prediction_accuracies_1)))
        #
        # confusion = np.array(confusion_matrix(y_true_overall, y_pred_overall)).astype("float")
        # total = np.sum(confusion, axis=1)
        # confusion = np.array([r/l for r, l in zip(confusion,total)])
        # logging.info("Confusion matrix \n%s" % confusion)
        # logging.info("Risk: %s" % (confusion[0,2]))
        #
        # logging.info("Overall accuracy, 2nd order: %s" % (np.mean(prediction_accuracies_2)))
        #
        # confusion = np.array(confusion_matrix(y_true_overall_2, y_pred_overall_2)).astype("float")
        # total = np.sum(confusion, axis=1)
        # confusion = np.array([r / l for r, l in zip(confusion, total)])
        # logging.info("Confusion matrix \n%s" % confusion)


        # logging.info("Risk: %s" % (confusion[0, 2]))



    def predict_next_state(self, initial_state, transition_matrix):
        this_trans = np.copy(transition_matrix)

        this_trans /= this_trans.sum(axis=1).reshape(-1, 1)
        this_trans[np.isnan(this_trans)] = 0

        trans_cumsum = np.cumsum(this_trans, axis=1)

        randomvalue = np.random.random()
        class_iter = 0

        if np.sum(trans_cumsum[initial_state]) == 0:
            print("Not transision possible, inital state: ", initial_state)
            return initial_state
        for pty in trans_cumsum[initial_state]:
            if randomvalue < pty:
                break

            class_iter += 1

        return class_iter

    def do_pred(self, time_step = 2):
        transision_matrix = self.load_model()

        import time
        start = time.time()
        pred_data, dates, prices = self.generate_pred_data(column="high")

        pred_classes = self.get_class(pred_data)

        end = time.time()
        logging.info(end - start)

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
        self.weeks_to_eval = args["weeks_to_eval"] if "weeks_to_eval" in args.keys() else 4

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

        total_samples_avail = max_items

        train_samples = values[:, :total_samples_avail]

        train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        return train_samples

    def generate_eval_data(self, column = "volume", stats = "max"):
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

        total_samples_avail = max_items

        train_samples = values[:, :total_samples_avail]

        train_samples = (train_samples[:,1:] - train_samples[:,0:-1])/train_samples[:,0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        return train_samples

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

    def do_eval(self):
        logging.info("Loading transition matrix")
        transision_matrix = self.load_model()

        logging.info("Generating eval data")
        eval_data_changes = self.generate_eval_data(column="high")
        logging.info("Generating eval data: Done")

        classes = self.get_classes(eval_data_changes)

        logging.info("Evaluating prediction accuracy")

        class_multiplier = [k for k in reversed([(self.n_classes) ** k for k in range(self.order)])]

        y_true_overall, y_true_overall_3 = [], []
        y_pred_overall, y_pred_overall_3 = [], []
        prediction_accuracies_1, prediction_accuracies_3 = [], []
        for ind, row in enumerate(classes):

            eval_data_for_this_row = row[-self.weeks_to_eval-1:]

            y_true = eval_data_for_this_row[1:]
            y_pred = []
            for k in eval_data_for_this_row[:-1]:

                initial_state = k
                y_pred += [self.predict_next_state(initial_state, transision_matrix[ind])]

            y_true_order_1 = [r for r in y_true]
            y_pred_order_1 = [r for r in y_pred]

            for k in class_multiplier[:-1]:
                y_true_order_1 = [r%k for r in y_true]
                y_pred_order_1 = [r%k for r in y_pred]


            prediction_accuracies_1 += [accuracy_score(y_true_order_1, y_pred_order_1)]
            logging.info("%s, 1st Order, GT: %s, Pred: %s, acc: %s" % (ind, y_true_order_1, y_pred_order_1, prediction_accuracies_1[-1]))

            y_true_overall += y_true_order_1
            y_pred_overall += y_pred_order_1

        logging.info("Overall accuracy, 1st order: %s" % (np.mean(prediction_accuracies_1)))

        confusion = np.array(confusion_matrix(y_true_overall, y_pred_overall)).astype("float")
        total = np.sum(confusion, axis=1)
        confusion = np.array([r/l for r, l in zip(confusion,total)])
        logging.info("Confusion matrix \n%s" % confusion)
        logging.info("Risk: %s" % (confusion[0,2]))


    def predict_next_state(self, initial_state, transition_matrix):
        this_trans = np.copy(transition_matrix)

        this_trans /= this_trans.sum(axis=1).reshape(-1, 1)
        this_trans[np.isnan(this_trans)] = 0

        trans_cumsum = np.cumsum(this_trans, axis=1)

        randomvalue = np.random.random()
        class_iter = 0

        if np.sum(trans_cumsum[initial_state]) == 0:
            print("Not transision possible, inital state: ", initial_state)
            return initial_state
        for pty in trans_cumsum[initial_state]:
            if randomvalue < pty:
                break

            class_iter += 1

        return class_iter

    def do_pred(self, time_step = 2):
        transision_matrix = self.load_model()

        import time
        start = time.time()
        pred_data, dates, prices = self.generate_pred_data(column="high")

        pred_classes = self.get_class(pred_data)

        end = time.time()
        logging.info(end - start)

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