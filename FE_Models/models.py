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
from keras.optimizers import RMSprop

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

class Cnn2ClassWeeklyVolumeLow(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = ['n_steps']
        self.opt_args = ['ouput_dir', "days_to_eval", "model_dir", 'pred_dir', 'epochs']
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

        self.blank_model = None

    def do_init(self, args):
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"

        self.model_dir = args[
            "model_dir"] if "model_dir" in args.keys() else self.output_dir + "training_dir/" + self.name + "/"
        self.eval_dir = args[
            "eval_dir"] if "eval_dir" in args.keys() else self.output_dir + "eval_dir/" + self.name + "/"
        self.pred_dir = args[
            "pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.model_file = os.path.join(os.path.dirname(self.model_dir), "model.npy")
        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")

        self.blank_filename = "blank.h5"
        self.blank_model = os.path.join(os.path.dirname(self.model_dir), self.blank_filename)

        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30
        self.weeks_to_eval = args["weeks_to_eval"] if "weeks_to_eval" in args.keys() else 8

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()
        self.company_list = self.db.get_list_companies()
        self.companies_count = self.db.get_companies_count()

        self.n_steps = args["n_steps"] if "n_steps" in args.keys() else 2
        self.epochs = args["epochs"] if "epochs" in args.keys() else 10
        self.n_classes = 2

        self.model = self.init_model()

    def init_model(self):

        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(3, self.n_steps), padding="same"))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding="same"))
        model.add(Dropout(0.50))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        # model.add(Dropout(0.25))
        # model.add(Dense(100, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))

        # model = Sequential()
        # model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(3, self.n_steps), padding="same"))
        # model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding="same"))
        # model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding="same"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.50))
        # model.add(MaxPooling1D(pool_size=2))
        # model.add(Flatten())
        # model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.25))
        # model.add(Dense(50, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.save_model(model, "blank")

        return model

    def save_model(self, model, stock):
        file_name = stock + ".h5"
        try:
            os.makedirs(self.model_dir)
        except OSError:
            logging.warning("Creation of the directory %s Falied (May already exist)" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        model.save_weights(os.path.join(self.model_dir, file_name))

        logging.info("Saving model %s", file_name)

    def load_model(self, stock):
        file_name = stock + ".h5"
        # with open(file_name, 'rb') as pkl:
        #     model = pickle.load(pkl)
        # return model
        logging.info("Loading model weights %s" % os.path.join(self.model_dir, file_name))
        self.model.load_weights(os.path.join(self.model_dir, file_name))

    def generate_train_data(self, column="volume", stats="max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = 0
        values_array, dates_array, volume_array, low_array = [], [], [], []

        for ind, k in enumerate(company_list):
            logging.info("Fetching Data from DB: Processing - %s/%s" % (ind, len(company_list)))
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]
            vol_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="volume", stats="mean")
            volume_array += [vol_temp]
            low_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="low", stats="min")
            low_array += [low_temp]

        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        values = np.zeros((total_companies, max_items))
        volume_values = np.zeros((total_companies, max_items))
        low_values =  np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

            low_fetch = low_array[i]
            low_values[i, max_items - len(values_fetch):max_items] = low_fetch

        total_samples_avail = max_items - self.weeks_to_eval

        train_samples = values[:, :total_samples_avail]
        train_samples = (train_samples[:, 1:] - train_samples[:, 0:-1]) / train_samples[:, 0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample = volume_values[:, :total_samples_avail]
        volume_sample = np.log((volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1])

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0

        low_sample = low_values[:, :total_samples_avail]
        low_sample = (low_sample[:, 1:] - low_sample[:, 0:-1]) / low_sample[:, 0:-1]

        low_sample[np.isnan(low_sample)] = 0
        low_sample[np.isinf(low_sample)] = 0

        return train_samples, volume_sample, low_values


    def generate_eval_data(self, column="volume", stats="max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = 0
        values_array, dates_array, volume_array, low_array = [], [], [], []

        for ind, k in enumerate(company_list):
            logging.info("Fetching Data from DB: Processing - %s/%s" % (ind, len(company_list)))
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]
            vol_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="volume", stats=stats)
            volume_array += [vol_temp]
            low_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="low", stats="min")
            low_array += [low_temp]

        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        values = np.zeros((total_companies, max_items))
        volume_values = np.zeros((total_companies, max_items))
        low_values = np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

            low_fetch = low_array[i]
            low_values[i, max_items - len(values_fetch):max_items] = low_fetch

        total_samples_avail = max_items

        train_samples = values[:, :total_samples_avail]
        train_samples = (train_samples[:, 1:] - train_samples[:, 0:-1]) / train_samples[:, 0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample = volume_values[:, :total_samples_avail]
        # volume_sample = np.log((volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1])
        volume_sample = np.log((volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1])

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0

        low_sample = low_values[:, :total_samples_avail]
        low_sample = (low_sample[:, 1:] - low_sample[:, 0:-1]) / low_sample[:, 0:-1]

        low_sample[np.isnan(low_sample)] = 0
        low_sample[np.isinf(low_sample)] = 0

        return train_samples, volume_sample, low_values

    def get_class_1stOrder(self, mat, labels={0: -0.001}):

        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)
        for k in range(len(labels)):
            if k == len(labels) - 1:
                continue
            matclasses[np.logical_and(mat > labels[k], mat <= labels[k + 1])] = k + 1
        else:
            matclasses[mat > labels[len(labels) - 1]] = len(labels)

        return matclasses


    def get_train_test(self, rows, rows_volume, rows_low, classes):
        x_train = []
        y_train = []
        for k in range(len(rows) - self.n_steps):
            x_train += [np.append(np.append(rows[k:k + self.n_steps], rows_volume[k:k + self.n_steps]), rows_low[k:k + self.n_steps])]

            y_train += [classes[k + self.n_steps]]
        x_train = np.array(x_train)

        x_train = np.reshape(x_train, (x_train.shape[0],3,-1))

        return x_train, y_train
        # return np.expand_dims(x_train, axis=3) , y_train

    def do_train(self):
        logging.info("Generating Train Data")
        train_data_changes, train_data_changes_volume, train_data_changes_low = self.generate_train_data(column="high")

        logging.info("Generating Train Data: Done")

        classes_train = self.get_class_1stOrder(train_data_changes)

        for k in classes_train:
            plt.hist(k)
            plt.show()

        logging.info("Training Global Model")
        # self.load_model("blank")
        # for r in range(self.epochs):
        #     epoch_acc = []
        #     for ind, k in enumerate(classes_train):
        #
        #         trimmed = np.trim_zeros(train_data_changes[ind])
        #         trimmed_volume = train_data_changes_volume[ind, -len(trimmed):]
        #         trimmed_low = train_data_changes_low[ind, -len(trimmed):]
        #         x_train, y_train = self.get_train_test(trimmed, trimmed_volume, trimmed_low, k[-len(trimmed):])
        #
        #         y_train = to_categorical(y_train, num_classes=self.n_classes)
        #
        #         metrics = self.model.fit(x_train, y_train, verbose=0)
        #         epoch_acc += [metrics.history['accuracy'][-1]]
        #         # logging.info("Training: Done, Accuracy: %s" % metrics.history['accuracy'][-1])
        #     logging.info("Epoch %s: Done, Accuracy: %s" % (r, np.mean(epoch_acc)))
        #
        # self.save_model(self.model, "global")
        # logging.info("Training Global Model: Done")

        logging.info("Fine tuning")

        for ind, k in enumerate(classes_train):
            self.load_model("blank")
            trimmed = np.trim_zeros(train_data_changes[ind])
            trimmed_volume = train_data_changes_volume[ind, -len(trimmed):]
            trimmed_low = train_data_changes_low[ind, -len(trimmed):]
            x_train, y_train = self.get_train_test(trimmed, trimmed_volume, trimmed_low, k[-len(trimmed):])

            y_train = to_categorical(y_train, num_classes=self.n_classes)

            metrics = self.model.fit(x_train, y_train, epochs=self.epochs, verbose=0)

            self.save_model(self.model, self.company_list[ind])

            logging.info("Fine tuning: Done, Accuracy: %s" % metrics.history['accuracy'][-1])

        # self.save_model(transition_matrix)

        return

    def do_eval(self):

        logging.info("Generating eval data")
        eval_data_changes, eval_data_changes_volume, eval_data_changes_low = self.generate_eval_data(column="high")
        logging.info("Generating eval data: Done")

        order_1_classes = self.get_class_1stOrder(eval_data_changes)

        logging.info("Evaluating prediction accuracy")
        prediction_accuracies = []

        y_true_overall = []
        y_pred_overall = []

        for ind, k in enumerate(order_1_classes):
            x_train, y_true = self.get_train_test(eval_data_changes[ind, -self.weeks_to_eval - self.n_steps:],
                                                  eval_data_changes_volume[ind, -self.weeks_to_eval - self.n_steps:],
                                                  eval_data_changes_low[ind, -self.weeks_to_eval - self.n_steps:],
                                                  k[-self.weeks_to_eval - self.n_steps:])

            self.load_model(self.company_list[ind])

            y_pred = self.model.predict(x_train)
            y_pred = np.argmax(y_pred, axis=1)

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

    def do_pred(self, time_step=2):
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

            predict_date = last_date + datetime.timedelta(days=7 * (t + 1))

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

class DeepCnn3ClassWeeklyVolumeLow(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = ['n_steps']
        self.opt_args = ['ouput_dir', "days_to_eval", "model_dir", 'pred_dir', 'epochs']
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

        self.blank_model = None

    def do_init(self, args):
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"

        self.model_dir = args[
            "model_dir"] if "model_dir" in args.keys() else self.output_dir + "training_dir/" + self.name + "/"
        self.eval_dir = args[
            "eval_dir"] if "eval_dir" in args.keys() else self.output_dir + "eval_dir/" + self.name + "/"
        self.pred_dir = args[
            "pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.model_file = os.path.join(os.path.dirname(self.model_dir), "model.npy")
        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")

        self.blank_filename = "blank.h5"
        self.blank_model = os.path.join(os.path.dirname(self.model_dir), self.blank_filename)

        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30
        self.weeks_to_eval = args["weeks_to_eval"] if "weeks_to_eval" in args.keys() else 8

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()
        self.company_list = self.db.get_list_companies()
        self.companies_count = self.db.get_companies_count()

        self.n_steps = args["n_steps"] if "n_steps" in args.keys() else 2
        self.epochs = args["epochs"] if "epochs" in args.keys() else 10
        self.n_classes = 3

        self.model = self.init_model()

    def init_model(self):

        # model = Sequential()
        # model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(3, self.n_steps), padding="same"))
        # model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding="same"))
        # model.add(Dropout(0.50))
        # model.add(MaxPooling1D(pool_size=2))
        # model.add(Flatten())
        # model.add(Dense(100, activation='relu'))
        # # model.add(Dropout(0.25))
        # # model.add(Dense(100, activation='relu'))
        # model.add(Dense(self.n_classes, activation='softmax'))

        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(3, self.n_steps, 1), padding="same"))
        model.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(3, self.n_steps, 1), padding="same"))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding="same"))
        model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding="same"))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding="same"))
        model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding="same"))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding="same"))
        model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding="same"))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=2))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.summary()
        
        opt = RMSprop()
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.save_model(model, "blank")

        return model

    def save_model(self, model, stock):
        file_name = stock + ".h5"
        try:
            os.makedirs(self.model_dir)
        except OSError:
            logging.warning("Creation of the directory %s Falied (May already exist)" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        model.save_weights(os.path.join(self.model_dir, file_name))

        logging.info("Saving model %s", file_name)

    def load_model(self, stock):
        file_name = stock + ".h5"
        # with open(file_name, 'rb') as pkl:
        #     model = pickle.load(pkl)
        # return model
        logging.info("Loading model weights %s" % os.path.join(self.model_dir, file_name))
        self.model.load_weights(os.path.join(self.model_dir, file_name))

    def generate_train_data(self, column="volume", stats="max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = 0
        values_array, dates_array, volume_array, low_array = [], [], [], []

        for ind, k in enumerate(company_list):
            logging.info("Fetching Data from DB: Processing - %s/%s" % (ind, len(company_list)))
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]
            vol_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="volume", stats="mean")
            volume_array += [vol_temp]
            low_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="low", stats="min")
            low_array += [low_temp]

        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        values = np.zeros((total_companies, max_items))
        volume_values = np.zeros((total_companies, max_items))
        low_values =  np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

            low_fetch = low_array[i]
            low_values[i, max_items - len(values_fetch):max_items] = low_fetch

        total_samples_avail = max_items - self.weeks_to_eval

        train_samples = values[:, :total_samples_avail]

        train_samples = (train_samples[:, 1:] - train_samples[:, 0:-1]) / train_samples[:, 0:-1]



        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample = volume_values[:, :total_samples_avail]
        volume_sample = np.log(1+(volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1])

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0

        low_sample = low_values[:, :total_samples_avail]
        low_sample = (low_sample[:, 1:] - low_sample[:, 0:-1]) / low_sample[:, 0:-1]

        low_sample[np.isnan(low_sample)] = 0
        low_sample[np.isinf(low_sample)] = 0

        return train_samples*100, volume_sample, low_values*100


    def generate_eval_data(self, column="volume", stats="max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = 0
        values_array, dates_array, volume_array, low_array = [], [], [], []

        for ind, k in enumerate(company_list):
            logging.info("Fetching Data from DB: Processing - %s/%s" % (ind, len(company_list)))
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]
            vol_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="volume", stats=stats)
            volume_array += [vol_temp]
            low_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="low", stats="min")
            low_array += [low_temp]

        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        values = np.zeros((total_companies, max_items))
        volume_values = np.zeros((total_companies, max_items))
        low_values = np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

            low_fetch = low_array[i]
            low_values[i, max_items - len(values_fetch):max_items] = low_fetch

        total_samples_avail = max_items

        train_samples = values[:, :total_samples_avail]
        train_samples = (train_samples[:, 1:] - train_samples[:, 0:-1]) / train_samples[:, 0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample = volume_values[:, :total_samples_avail]
        # volume_sample = np.log((volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1])
        volume_sample = np.log(1+(volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1])

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0

        low_sample = low_values[:, :total_samples_avail]
        low_sample = (low_sample[:, 1:] - low_sample[:, 0:-1]) / low_sample[:, 0:-1]

        low_sample[np.isnan(low_sample)] = 0
        low_sample[np.isinf(low_sample)] = 0

        return train_samples*100, volume_sample, low_values*100

    def get_class_1stOrder(self, mat, labels={0: -1, 1: 1}):

        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)
        for k in range(len(labels)):
            if k == len(labels) - 1:
                continue
            matclasses[np.logical_and(mat > labels[k], mat <= labels[k + 1])] = k + 1
        else:
            matclasses[mat > labels[len(labels) - 1]] = len(labels)

        return matclasses


    def get_train_test(self, rows, rows_volume, rows_low, classes):
        x_train = []
        y_train = []
        for k in range(len(rows) - self.n_steps):
            x_train += [np.append(np.append(rows[k:k + self.n_steps], rows_volume[k:k + self.n_steps]), rows_low[k:k + self.n_steps])]

            y_train += [classes[k + self.n_steps]]
        x_train = np.array(x_train)

        x_train = np.reshape(x_train, (x_train.shape[0],3,-1))

        return x_train, y_train
        # return np.expand_dims(x_train, axis=3) , y_train

    # def get_even_distributed_train_data(self, x_train, y_train):
    #
    #     x_train_0 = x_train[y_train == 0]
    #     y_train_0 = y_train[y_train == 0]
    #     x_train_1 = x_train[y_train == 1]
    #     y_train_1 = y_train[y_train == 1]
    #     x_train_2 = x_train[y_train == 2]
    #     y_train_2 = y_train[y_train == 2]
    #
    #     print(y_train_0.shape, y_train_1.shape, y_train_2.shape)
    #     exit()

    def do_train(self):
        logging.info("Generating Train Data")
        train_data_changes, train_data_changes_volume, train_data_changes_low = self.generate_train_data(column="high")

        logging.info("Generating Train Data: Done")

        classes_train = self.get_class_1stOrder(train_data_changes)

        # for ind, k in enumerate(classes_train):
        #     trimmed = np.trim_zeros(train_data_changes[ind])
        #     plt.hist(k[-len(trimmed):])
        #     plt.show()

        logging.info("Training Global Model")
        self.load_model("blank")

        logging.info("Loading training data")
        x_train, y_train = None, None
        for ind in range(classes_train.shape[0]):
            trimmed = np.trim_zeros(train_data_changes[ind])
            trimmed_volume = train_data_changes_volume[ind, -len(trimmed):]
            trimmed_low = train_data_changes_low[ind, -len(trimmed):]
            x_train_, y_train_ = self.get_train_test(trimmed, trimmed_volume, trimmed_low,
                                                   classes_train[ind, -len(trimmed):])
            if x_train is None:
                x_train = x_train_
            else:
                x_train = np.append(x_train, x_train_, axis=0)

            if y_train is None:
                y_train = y_train_
            else:
                y_train = np.append(y_train, y_train_)


        # for r in range(self.epochs):
        #     epoch_acc = []
        # x_train = x_train/np.max(x_train, axis=0)

        y_train = to_categorical(y_train, num_classes=self.n_classes)
        x_train = np.expand_dims(x_train, axis=3)
        for k in range(self.epochs):
            print("epoch: %s" % k)
            x_train_a = x_train + np.random.normal(0, 0.01, x_train.shape)
            metrics = self.model.fit(x_train_a, y_train, verbose=1, epochs=1, batch_size=32)
        # epoch_acc += [metrics.history['accuracy'][-1]]
        # logging.info("Training: Done, Accuracy: %s" % metrics.history['accuracy'][-1])
        # logging.info("Epoch %s: Done, Accuracy: %s" % (r, np.mean(epoch_acc)))

        self.save_model(self.model, "global")
        logging.info("Training Global Model: Done")

        # logging.info("Fine tuning")
        #
        # for ind, k in enumerate(classes_train):
        #     self.load_model("blank")
        #     trimmed = np.trim_zero(train_data_changes[ind])
        #     trimmed_volume = train_data_changes_volume[ind, -len(trimmed):]
        #     trimmed_low = train_data_changes_low[ind, -len(trimmed):]
        #     x_train, y_train = self.get_train_test(trimmed, trimmed_volume, trimmed_low, k[-len(trimmed):])
        #
        #     y_train = to_categorical(y_train, num_classes=self.n_classes)
        #
        #     metrics = self.model.fit(x_train, y_train, epochs=self.epochs, verbose=0)
        #
        #     self.save_model(self.model, self.company_list[ind])
        #
        #     logging.info("Fine tuning: Done, Accuracy: %s" % metrics.history['accuracy'][-1])

        # self.save_model(transition_matrix)

        return

    def do_eval(self):

        logging.info("Generating eval data")
        eval_data_changes, eval_data_changes_volume, eval_data_changes_low = self.generate_eval_data(column="high")
        logging.info("Generating eval data: Done")

        order_1_classes = self.get_class_1stOrder(eval_data_changes)

        logging.info("Evaluating prediction accuracy")
        prediction_accuracies = []

        y_true_overall = []
        y_pred_overall = []

        for ind, k in enumerate(order_1_classes):
            x_train, y_true = self.get_train_test(eval_data_changes[ind, -self.weeks_to_eval - self.n_steps:],
                                                  eval_data_changes_volume[ind, -self.weeks_to_eval - self.n_steps:],
                                                  eval_data_changes_low[ind, -self.weeks_to_eval - self.n_steps:],
                                                  k[-self.weeks_to_eval - self.n_steps:])

            # logging.info(x_train.shape)

            # new_true = []
            # new_z = []
            # for ind, filter in enumerate(y_true):
            #     if filter == 0:
            #         if ind < len(y_true)-1:
            #             new_true += [y_true[ind+1]]
            #             new_z += [x_train[ind+1]]
            #             # logging.info(np.array(new_z).shape)
            # if not new_true:
            #     continue
            #
            # y_true = new_true
            # x_train = new_z



            # self.load_model(self.company_list[ind])
            self.load_model("global")
            x_train = np.expand_dims(x_train, axis=3)
            y_pred = self.model.predict(x_train)
            y_pred = np.argmax(y_pred, axis=1)

            prediction_accuracies += [accuracy_score(y_true, y_pred)]
            logging.info("%s, 1st Order, GT: %s, Pred: %s, acc: %s" % (ind, y_true, y_pred, prediction_accuracies[-1]))

            y_true_overall += y_true
            y_pred_overall += [y_pred]


        y_true_overall = np.array(y_true_overall).flatten()
        new_r = []
        for k in y_pred_overall:
            for l in k:
                new_r += [l]
        y_pred_overall = np.array(new_r)


        logging.info("Overall accuracy, 1st order: %s" % (np.mean(prediction_accuracies)))
        print(y_true_overall, y_pred_overall)
        print(y_true_overall.shape, y_pred_overall.shape)
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

    def do_pred(self, time_step=2):
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

            predict_date = last_date + datetime.timedelta(days=7 * (t + 1))

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

class Cnn3ClassWeeklyVolumeLow(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = ['n_steps']
        self.opt_args = ['ouput_dir', "days_to_eval", "model_dir", 'pred_dir', 'epochs']
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

        self.blank_model = None

    def do_init(self, args):
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"

        self.model_dir = args[
            "model_dir"] if "model_dir" in args.keys() else self.output_dir + "training_dir/" + self.name + "/"
        self.eval_dir = args[
            "eval_dir"] if "eval_dir" in args.keys() else self.output_dir + "eval_dir/" + self.name + "/"
        self.pred_dir = args[
            "pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.model_file = os.path.join(os.path.dirname(self.model_dir), "model.npy")
        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")

        self.blank_filename = "blank.h5"
        self.blank_model = os.path.join(os.path.dirname(self.model_dir), self.blank_filename)

        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30
        self.weeks_to_eval = args["weeks_to_eval"] if "weeks_to_eval" in args.keys() else 8

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()
        self.company_list = self.db.get_list_companies()
        self.companies_count = self.db.get_companies_count()

        self.n_steps = args["n_steps"] if "n_steps" in args.keys() else 2
        self.epochs = args["epochs"] if "epochs" in args.keys() else 10
        self.n_classes = 3

        self.model = self.init_model()

    def init_model(self):

        # model = Sequential()
        # model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(3, self.n_steps), padding="same"))
        # model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding="same"))
        # model.add(Dropout(0.50))
        # model.add(MaxPooling1D(pool_size=2))
        # model.add(Flatten())
        # model.add(Dense(100, activation='relu'))
        # # model.add(Dropout(0.25))
        # # model.add(Dense(100, activation='relu'))
        # model.add(Dense(self.n_classes, activation='softmax'))

        model = Sequential()
        model.add(Conv2D(filters=128, kernel_size=3, activation='relu', input_shape=(3, self.n_steps, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding="same"))
        model.add(BatchNormalization())

        model.add(Dropout(0.50))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.summary()
        opt = RMSprop()
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.save_model(model, "blank")

        return model

    def save_model(self, model, stock):
        file_name = stock + ".h5"
        try:
            os.makedirs(self.model_dir)
        except OSError:
            logging.warning("Creation of the directory %s Falied (May already exist)" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        model.save_weights(os.path.join(self.model_dir, file_name))

        logging.info("Saving model %s", file_name)

    def load_model(self, stock):
        file_name = stock + ".h5"
        # with open(file_name, 'rb') as pkl:
        #     model = pickle.load(pkl)
        # return model
        logging.info("Loading model weights %s" % os.path.join(self.model_dir, file_name))
        self.model.load_weights(os.path.join(self.model_dir, file_name))

    def generate_train_data(self, column="volume", stats="max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = 0
        values_array, dates_array, volume_array, low_array = [], [], [], []

        for ind, k in enumerate(company_list):
            logging.info("Fetching Data from DB: Processing - %s/%s" % (ind, len(company_list)))
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]
            vol_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="volume", stats="mean")
            volume_array += [vol_temp]
            low_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="low", stats="min")
            low_array += [low_temp]

        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        values = np.zeros((total_companies, max_items))
        volume_values = np.zeros((total_companies, max_items))
        low_values =  np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

            low_fetch = low_array[i]
            low_values[i, max_items - len(values_fetch):max_items] = low_fetch

        total_samples_avail = max_items - self.weeks_to_eval

        train_samples = values[:, :total_samples_avail]

        train_samples = (train_samples[:, 1:] - train_samples[:, 0:-1]) / train_samples[:, 0:-1]



        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample = volume_values[:, :total_samples_avail]
        volume_sample = np.log(1+(volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1])

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0

        low_sample = low_values[:, :total_samples_avail]
        low_sample = (low_sample[:, 1:] - low_sample[:, 0:-1]) / low_sample[:, 0:-1]

        low_sample[np.isnan(low_sample)] = 0
        low_sample[np.isinf(low_sample)] = 0

        return train_samples*100, volume_sample, low_values*100


    def generate_eval_data(self, column="volume", stats="max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = 0
        values_array, dates_array, volume_array, low_array = [], [], [], []

        for ind, k in enumerate(company_list):
            logging.info("Fetching Data from DB: Processing - %s/%s" % (ind, len(company_list)))
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]
            vol_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="volume", stats=stats)
            volume_array += [vol_temp]
            low_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="low", stats="min")
            low_array += [low_temp]

        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        values = np.zeros((total_companies, max_items))
        volume_values = np.zeros((total_companies, max_items))
        low_values = np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

            low_fetch = low_array[i]
            low_values[i, max_items - len(values_fetch):max_items] = low_fetch

        total_samples_avail = max_items

        train_samples = values[:, :total_samples_avail]
        train_samples = (train_samples[:, 1:] - train_samples[:, 0:-1]) / train_samples[:, 0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample = volume_values[:, :total_samples_avail]
        # volume_sample = np.log((volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1])
        volume_sample = np.log(1+(volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1])

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0

        low_sample = low_values[:, :total_samples_avail]
        low_sample = (low_sample[:, 1:] - low_sample[:, 0:-1]) / low_sample[:, 0:-1]

        low_sample[np.isnan(low_sample)] = 0
        low_sample[np.isinf(low_sample)] = 0

        return train_samples*100, volume_sample, low_values*100

    def get_class_1stOrder(self, mat, labels={0: -1, 1: 1}):

        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)
        for k in range(len(labels)):
            if k == len(labels) - 1:
                continue
            matclasses[np.logical_and(mat > labels[k], mat <= labels[k + 1])] = k + 1
        else:
            matclasses[mat > labels[len(labels) - 1]] = len(labels)

        return matclasses


    def get_train_test(self, rows, rows_volume, rows_low, classes):
        x_train = []
        y_train = []
        for k in range(len(rows) - self.n_steps):
            x_train += [np.append(np.append(rows[k:k + self.n_steps], rows_volume[k:k + self.n_steps]), rows_low[k:k + self.n_steps])]

            y_train += [classes[k + self.n_steps]]
        x_train = np.array(x_train)

        x_train = np.reshape(x_train, (x_train.shape[0],3,-1))

        return x_train, y_train
        # return np.expand_dims(x_train, axis=3) , y_train

    # def get_even_distributed_train_data(self, x_train, y_train):
    #
    #     x_train_0 = x_train[y_train == 0]
    #     y_train_0 = y_train[y_train == 0]
    #     x_train_1 = x_train[y_train == 1]
    #     y_train_1 = y_train[y_train == 1]
    #     x_train_2 = x_train[y_train == 2]
    #     y_train_2 = y_train[y_train == 2]
    #
    #     print(y_train_0.shape, y_train_1.shape, y_train_2.shape)
    #     exit()

    def do_train(self):
        logging.info("Generating Train Data")
        train_data_changes, train_data_changes_volume, train_data_changes_low = self.generate_train_data(column="high")

        logging.info("Generating Train Data: Done")

        classes_train = self.get_class_1stOrder(train_data_changes)

        # for ind, k in enumerate(classes_train):
        #     trimmed = np.trim_zeros(train_data_changes[ind])
        #     plt.hist(k[-len(trimmed):])
        #     plt.show()

        logging.info("Training Global Model")
        self.load_model("blank")

        logging.info("Loading training data")
        x_train, y_train = None, None
        for ind in range(classes_train.shape[0]):
            trimmed = np.trim_zeros(train_data_changes[ind])
            trimmed_volume = train_data_changes_volume[ind, -len(trimmed):]
            trimmed_low = train_data_changes_low[ind, -len(trimmed):]
            x_train_, y_train_ = self.get_train_test(trimmed, trimmed_volume, trimmed_low,
                                                   classes_train[ind, -len(trimmed):])
            if x_train is None:
                x_train = x_train_
            else:
                x_train = np.append(x_train, x_train_, axis=0)

            if y_train is None:
                y_train = y_train_
            else:
                y_train = np.append(y_train, y_train_)


        # for r in range(self.epochs):
        #     epoch_acc = []
        # x_train = x_train/np.max(x_train, axis=0)

        y_train = to_categorical(y_train, num_classes=self.n_classes)
        x_train = np.expand_dims(x_train, axis=3)
        for k in range(self.epochs):
            print("epoch: %s" % k)
            x_train_a = x_train + np.random.normal(0, 0.01, x_train.shape)
            metrics = self.model.fit(x_train_a, y_train, verbose=1, epochs=1, batch_size=32)
        # epoch_acc += [metrics.history['accuracy'][-1]]
        # logging.info("Training: Done, Accuracy: %s" % metrics.history['accuracy'][-1])
        # logging.info("Epoch %s: Done, Accuracy: %s" % (r, np.mean(epoch_acc)))

        self.save_model(self.model, "global")
        logging.info("Training Global Model: Done")

        # logging.info("Fine tuning")
        #
        # for ind, k in enumerate(classes_train):
        #     self.load_model("blank")
        #     trimmed = np.trim_zero(train_data_changes[ind])
        #     trimmed_volume = train_data_changes_volume[ind, -len(trimmed):]
        #     trimmed_low = train_data_changes_low[ind, -len(trimmed):]
        #     x_train, y_train = self.get_train_test(trimmed, trimmed_volume, trimmed_low, k[-len(trimmed):])
        #
        #     y_train = to_categorical(y_train, num_classes=self.n_classes)
        #
        #     metrics = self.model.fit(x_train, y_train, epochs=self.epochs, verbose=0)
        #
        #     self.save_model(self.model, self.company_list[ind])
        #
        #     logging.info("Fine tuning: Done, Accuracy: %s" % metrics.history['accuracy'][-1])

        # self.save_model(transition_matrix)

        return

    def do_eval(self):

        logging.info("Generating eval data")
        eval_data_changes, eval_data_changes_volume, eval_data_changes_low = self.generate_eval_data(column="high")
        logging.info("Generating eval data: Done")

        order_1_classes = self.get_class_1stOrder(eval_data_changes)

        logging.info("Evaluating prediction accuracy")
        prediction_accuracies = []

        y_true_overall = []
        y_pred_overall = []

        for ind, k in enumerate(order_1_classes):
            x_train, y_true = self.get_train_test(eval_data_changes[ind, -self.weeks_to_eval - self.n_steps:],
                                                  eval_data_changes_volume[ind, -self.weeks_to_eval - self.n_steps:],
                                                  eval_data_changes_low[ind, -self.weeks_to_eval - self.n_steps:],
                                                  k[-self.weeks_to_eval - self.n_steps:])

            # logging.info(x_train.shape)

            # new_true = []
            # new_z = []
            # for ind, filter in enumerate(y_true):
            #     if filter == 0:
            #         if ind < len(y_true)-1:
            #             new_true += [y_true[ind+1]]
            #             new_z += [x_train[ind+1]]
            #             # logging.info(np.array(new_z).shape)
            # if not new_true:
            #     continue
            #
            # y_true = new_true
            # x_train = new_z



            # self.load_model(self.company_list[ind])
            self.load_model("global")
            x_train = np.expand_dims(x_train, axis=3)
            y_pred = self.model.predict(x_train)
            y_pred = np.argmax(y_pred, axis=1)

            prediction_accuracies += [accuracy_score(y_true, y_pred)]
            logging.info("%s, 1st Order, GT: %s, Pred: %s, acc: %s" % (ind, y_true, y_pred, prediction_accuracies[-1]))

            y_true_overall += y_true
            y_pred_overall += [y_pred]


        y_true_overall = np.array(y_true_overall).flatten()
        new_r = []
        for k in y_pred_overall:
            for l in k:
                new_r += [l]
        y_pred_overall = np.array(new_r)


        logging.info("Overall accuracy, 1st order: %s" % (np.mean(prediction_accuracies)))
        print(y_true_overall, y_pred_overall)
        print(y_true_overall.shape, y_pred_overall.shape)
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

    def do_pred(self, time_step=2):
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

            predict_date = last_date + datetime.timedelta(days=7 * (t + 1))

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


class Resnet3ClassWeeklyVolumeLow(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = ['n_steps']
        self.opt_args = ['ouput_dir', "days_to_eval", "model_dir", 'pred_dir', 'epochs']
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

        self.blank_model = None

        self.epochs = None

    def do_init(self, args):
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"

        self.model_dir = args[
            "model_dir"] if "model_dir" in args.keys() else self.output_dir + "training_dir/" + self.name + "/"
        self.eval_dir = args[
            "eval_dir"] if "eval_dir" in args.keys() else self.output_dir + "eval_dir/" + self.name + "/"
        self.pred_dir = args[
            "pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.model_file = os.path.join(os.path.dirname(self.model_dir), "model.npy")
        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")

        self.blank_filename = "blank.h5"
        self.blank_model = os.path.join(os.path.dirname(self.model_dir), self.blank_filename)

        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30
        self.weeks_to_eval = args["weeks_to_eval"] if "weeks_to_eval" in args.keys() else 4

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()
        self.company_list = self.db.get_list_companies()
        self.companies_count = self.db.get_companies_count()

        self.n_steps = args["n_steps"] if "n_steps" in args.keys() else 2
        self.n_classes = 3

        self.epochs = args["epochs"] if "epochs" in args.keys() else 50

        self.model = self.init_model()

    def init_model(self):
        n_strides = 1
        n_filters = 64
        n_pool = 2
        n_kernel = 3
        drop_rate = 0.5
        x = Input(shape=(3, self.n_steps))

        # First Conv / BN / ReLU layer
        y = Conv1D(filters=n_filters, kernel_size=3, strides=n_strides, padding='same')(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        # shortcut = y
        shortcut = MaxPooling1D(pool_size=n_pool)(y)
        # shortcut = AveragePooling1D(pool_size=n_pool)(y)

        # First Residual block
        y = Conv1D(filters=n_filters, kernel_size=n_kernel, strides=n_strides, padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Dropout(rate=drop_rate)(y)
        y = Conv1D(filters=n_filters, kernel_size=n_kernel, strides=n_strides, padding='same')(y)

        # Add Residual (shortcut)
        y = Add()([shortcut, y])

        # Repeated Residual blocks
        for k in range(2, 5):  # smaller network for testing
            # shortcut = y
            shortcut = MaxPooling1D(pool_size=n_pool)(y)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Dropout(rate=drop_rate)(y)
            y = Conv1D(filters=n_filters * k, kernel_size=n_kernel, strides=n_strides, padding='same')(y)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Dropout(rate=drop_rate)(y)
            y = Conv1D(filters=n_filters, kernel_size=n_kernel, strides=n_strides, padding='same')(y)
            # print(y, shortcut)
            y = Add()([shortcut, y])

        z = BatchNormalization()(y)
        z = Activation('relu')(z)
        z = Flatten()(z)
        z = Dense(64, activation='relu')(z)
        predictions = Dense(self.n_classes, activation='softmax')(z)

        model = Model(inputs=x, outputs=predictions)

        # Compiling
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        self.save_model(model, "blank")
        return model

    def save_model(self, model, stock):
        file_name = stock + ".h5"
        try:
            os.makedirs(self.model_dir)
        except OSError:
            logging.warning("Creation of the directory %s Falied (May already exist)" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        model.save_weights(os.path.join(self.model_dir, file_name))

        logging.info("Saving model %s", file_name)

    def load_model(self, stock):
        file_name = stock + ".h5"
        # with open(file_name, 'rb') as pkl:
        #     model = pickle.load(pkl)
        # return model
        logging.info("Loading model weights %s" % os.path.join(self.model_dir, file_name))
        self.model.load_weights(os.path.join(self.model_dir, file_name))

    def generate_train_data(self, column="volume", stats="max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = 0
        values_array, dates_array, volume_array, low_array = [], [], [], []

        for ind, k in enumerate(company_list):
            logging.info("Fetching Data from DB: Processing - %s/%s" % (ind, len(company_list)))
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]
            vol_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="volume", stats="mean")
            volume_array += [vol_temp]
            low_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="low", stats="min")
            low_array += [low_temp]

        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        values = np.zeros((total_companies, max_items))
        volume_values = np.zeros((total_companies, max_items))
        low_values =  np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

            low_fetch = low_array[i]
            low_values[i, max_items - len(values_fetch):max_items] = low_fetch

        total_samples_avail = max_items - self.weeks_to_eval

        train_samples = values[:, :total_samples_avail]
        train_samples = (train_samples[:, 1:] - train_samples[:, 0:-1]) / train_samples[:, 0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample = volume_values[:, :total_samples_avail]
        volume_sample = np.log((volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1])

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0

        low_sample = low_values[:, :total_samples_avail]
        low_sample = (low_sample[:, 1:] - low_sample[:, 0:-1]) / low_sample[:, 0:-1]

        low_sample[np.isnan(low_sample)] = 0
        low_sample[np.isinf(low_sample)] = 0

        return train_samples, volume_sample, low_values


    def generate_eval_data(self, column="volume", stats="max"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = 0
        values_array, dates_array, volume_array, low_array = [], [], [], []

        for ind, k in enumerate(company_list):
            logging.info("Fetching Data from DB: Processing - %s/%s" % (ind, len(company_list)))
            v_temp, d_temp = self.db.get_weekly_stats_company(company_sym=k, columns=column, stats=stats)
            values_array += [v_temp]
            dates_array += [d_temp]
            vol_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="volume", stats=stats)
            volume_array += [vol_temp]
            low_temp, _ = self.db.get_weekly_stats_company(company_sym=k, columns="low", stats="min")
            low_array += [low_temp]

        for k in dates_array:
            if max_items < len(k):
                max_items = len(k)

        values = np.zeros((total_companies, max_items))
        volume_values = np.zeros((total_companies, max_items))
        low_values = np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

            low_fetch = low_array[i]
            low_values[i, max_items - len(values_fetch):max_items] = low_fetch

        total_samples_avail = max_items

        train_samples = values[:, :total_samples_avail]
        train_samples = (train_samples[:, 1:] - train_samples[:, 0:-1]) / train_samples[:, 0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample = volume_values[:, :total_samples_avail]
        volume_sample = np.log((volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1])

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0

        low_sample = low_values[:, :total_samples_avail]
        low_sample = (low_sample[:, 1:] - low_sample[:, 0:-1]) / low_sample[:, 0:-1]

        low_sample[np.isnan(low_sample)] = 0
        low_sample[np.isinf(low_sample)] = 0

        return train_samples, volume_sample, low_values

    def get_class_1stOrder(self, mat, labels={0: -0.01, 1: 0.01}):

        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)
        for k in range(len(labels)):
            if k == len(labels) - 1:
                continue
            matclasses[np.logical_and(mat > labels[k], mat <= labels[k + 1])] = k + 1
        else:
            matclasses[mat > labels[len(labels) - 1]] = len(labels)

        return matclasses


    def get_train_test(self, rows, rows_volume, rows_low, classes):
        x_train = []
        y_train = []
        for k in range(len(rows) - self.n_steps):
            x_train += [np.append(np.append(rows[k:k + self.n_steps], rows_volume[k:k + self.n_steps]), rows_low[k:k + self.n_steps])]

            y_train += [classes[k + self.n_steps]]
        x_train = np.array(x_train)

        x_train = np.reshape(x_train, (x_train.shape[0],3,-1))

        return x_train, y_train
        # return np.expand_dims(x_train, axis=3) , y_train

    def do_train(self):
        logging.info("Generating Train Data")
        train_data_changes, train_data_changes_volume, train_data_changes_low = self.generate_train_data(column="high")

        logging.info("Generating Train Data: Done")

        classes_train = self.get_class_1stOrder(train_data_changes)
        logging.info("Training Global Model")
        logging.info("Total Epocs: %s" % self.epochs)
        self.load_model("blank")

        for r in range(self.epochs):
            epoch_acc = []
            indices = [k for k in range(classes_train.shape[0])]
            np.random.shuffle(indices)
            for ind in indices:

                trimmed = np.trim_zeros(train_data_changes[ind])
                trimmed_volume = train_data_changes_volume[ind, -len(trimmed):]
                trimmed_low = train_data_changes_low[ind, -len(trimmed):]
                x_train, y_train = self.get_train_test(trimmed, trimmed_volume, trimmed_low, classes_train[ind, -len(trimmed):])

                y_train = to_categorical(y_train, num_classes=self.n_classes)

                metrics = self.model.fit(x_train, y_train, verbose=0)
                epoch_acc += [metrics.history['accuracy'][-1]]
                # logging.info("Training: Done, Accuracy: %s" % metrics.history['accuracy'][-1])
            logging.info("Epoch %s: Done, Accuracy: %s" % (r, np.mean(epoch_acc)))

        self.save_model(self.model, "global")
        logging.info("Training Global Model: Done")

        logging.info("Fine tuning")

        # for ind, k in enumerate(classes_train):
        #     self.load_model("blank")
        #     trimmed = np.trim_zeros(train_data_changes[ind])
        #     trimmed_volume = train_data_changes_volume[ind, -len(trimmed):]
        #     trimmed_low = train_data_changes_low[ind, -len(trimmed):]
        #     x_train, y_train = self.get_train_test(trimmed, trimmed_volume, trimmed_low, k[-len(trimmed):])
        #
        #     y_train = to_categorical(y_train, num_classes=self.n_classes)
        #
        #     metrics = self.model.fit(x_train, y_train, epochs=100, verbose=0)
        #
        #     self.save_model(self.model, self.company_list[ind])
        #
        #     logging.info("Fine tuning: Done, Accuracy: %s" % metrics.history['accuracy'][-1])

        # self.save_model(transition_matrix)

        return

    def do_eval(self):

        logging.info("Generating eval data")
        eval_data_changes, eval_data_changes_volume, eval_data_changes_low = self.generate_eval_data(column="high")
        logging.info("Generating eval data: Done")

        order_1_classes = self.get_class_1stOrder(eval_data_changes)

        logging.info("Evaluating prediction accuracy")
        prediction_accuracies = []

        y_true_overall = []
        y_pred_overall = []

        for ind, k in enumerate(order_1_classes):
            x_train, y_true = self.get_train_test(eval_data_changes[ind, -self.weeks_to_eval - self.n_steps:],
                                                  eval_data_changes_volume[ind, -self.weeks_to_eval - self.n_steps:],
                                                  eval_data_changes_low[ind, -self.weeks_to_eval - self.n_steps:],
                                                  k[-self.weeks_to_eval - self.n_steps:])

            # self.load_model(self.company_list[ind])
            self.load_model("global")

            y_pred = self.model.predict(x_train)
            y_pred = np.argmax(y_pred, axis=1)

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

    def do_pred(self, time_step=2):
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

            predict_date = last_date + datetime.timedelta(days=7 * (t + 1))

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


class Cnn3ClassWeeklyVolume(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = ["n_steps"]
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

        self.blank_model = None

    def do_init(self, args):
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"

        self.model_dir = args[
            "model_dir"] if "model_dir" in args.keys() else self.output_dir + "training_dir/" + self.name + "/"
        self.eval_dir = args[
            "eval_dir"] if "eval_dir" in args.keys() else self.output_dir + "eval_dir/" + self.name + "/"
        self.pred_dir = args[
            "pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.model_file = os.path.join(os.path.dirname(self.model_dir), "model.npy")
        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")

        self.blank_filename = "blank.h5"
        self.blank_model = os.path.join(os.path.dirname(self.model_dir), self.blank_filename)

        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30
        self.weeks_to_eval = args["weeks_to_eval"] if "weeks_to_eval" in args.keys() else 8

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()
        self.company_list = self.db.get_list_companies()
        self.companies_count = self.db.get_companies_count()

        self.n_steps = args["n_steps"] if "n_steps" in args.keys() else 2
        self.n_classes = 3

        self.model = self.init_model()

    def init_model(self):

        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(2, self.n_steps), padding="same"))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding="same"))
        model.add(Dropout(0.50))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        # model.add(Dropout(0.25))
        # model.add(Dense(50, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.save_model(model, "blank")

        return model

    def save_model(self, model, stock):
        file_name = stock + ".h5"
        try:
            os.makedirs(self.model_dir)
        except OSError:
            logging.warning("Creation of the directory %s Falied (May already exist)" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        model.save_weights(os.path.join(self.model_dir, file_name))

        logging.info("Saving model %s", file_name)

    def load_model(self, stock):
        file_name = stock + ".h5"
        # with open(file_name, 'rb') as pkl:
        #     model = pickle.load(pkl)
        # return model
        logging.info("Loading model weights %s" % os.path.join(self.model_dir, file_name))
        self.model.load_weights(os.path.join(self.model_dir, file_name))

    def generate_train_data(self, column="volume", stats="max"):
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
        volume_values = np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

        total_samples_avail = max_items - self.weeks_to_eval

        train_samples = values[:, :total_samples_avail]
        train_samples = (train_samples[:, 1:] - train_samples[:, 0:-1]) / train_samples[:, 0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample = volume_values[:, :total_samples_avail]
        volume_sample = (volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1]

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0

        return train_samples, volume_sample

    def generate_eval_data(self, column="volume", stats="max"):
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
        volume_values = np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

        total_samples_avail = max_items

        train_samples = values[:, :total_samples_avail]
        train_samples = (train_samples[:, 1:] - train_samples[:, 0:-1]) / train_samples[:, 0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample = volume_values[:, :total_samples_avail]
        volume_sample = (volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1]

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0

        return train_samples, volume_sample

    def get_class_1stOrder(self, mat, labels={0: -0.01, 1: 0.01}):

        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)
        for k in range(len(labels)):
            if k == len(labels) - 1:
                continue
            matclasses[np.logical_and(mat > labels[k], mat <= labels[k + 1])] = k + 1
        else:
            matclasses[mat > labels[len(labels) - 1]] = len(labels)

        return matclasses


    def get_train_test(self, rows, rows_volume, classes):
        x_train = []
        y_train = []
        for k in range(len(rows) - self.n_steps):
            x_train += [np.append(rows[k:k + self.n_steps], rows_volume[k:k + self.n_steps])]

            y_train += [classes[k + self.n_steps]]
        x_train = np.array(x_train)

        x_train = np.reshape(x_train, (x_train.shape[0],2,-1))

        return x_train, y_train
        # return np.expand_dims(x_train, axis=3) , y_train

    def do_train(self):
        logging.info("Generating Train Data")
        train_data_changes, train_data_changes_volume = self.generate_train_data(column="high")

        logging.info("Generating Train Data: Done")

        classes_train = self.get_class_1stOrder(train_data_changes)
        # logging.info("Training Global Model")
        # self.load_model("blank")
        # for r in range(50):
        #     epoch_acc = []
        #     for ind, k in enumerate(classes_train):
        #
        #         trimmed = np.trim_zeros(train_data_changes[ind])
        #         trimmed_volume = train_data_changes_volume[ind, -len(trimmed):]
        #         x_train, y_train = self.get_train_test(trimmed, trimmed_volume, k[-len(trimmed):])
        #
        #         y_train = to_categorical(y_train, num_classes=self.n_classes)
        #
        #         metrics = self.model.fit(x_train, y_train, verbose=0)
        #         epoch_acc += [metrics.history['accuracy'][-1]]
        #         # logging.info("Training: Done, Accuracy: %s" % metrics.history['accuracy'][-1])
        #     logging.info("Epoch %s: Done, Accuracy: %s" % (r, np.mean(epoch_acc)))
        #
        # self.save_model(self.model, "global")
        # logging.info("Training Global Model: Done")

        logging.info("Trianing")

        for ind, k in enumerate(classes_train):
            self.load_model("blank")
            trimmed = np.trim_zeros(train_data_changes[ind])
            trimmed_volume = train_data_changes_volume[ind, -len(trimmed):]
            x_train, y_train = self.get_train_test(trimmed, trimmed_volume, k[-len(trimmed):])

            y_train = to_categorical(y_train, num_classes=self.n_classes)

            metrics = self.model.fit(x_train, y_train, epochs=100, verbose=0)

            self.save_model(self.model, self.company_list[ind])

            logging.info("Trianing: Done, Accuracy: %s" % metrics.history['accuracy'][-1])


        return

    def do_eval(self):

        logging.info("Generating eval data")
        eval_data_changes, eval_data_changes_volume = self.generate_eval_data(column="high")
        logging.info("Generating eval data: Done")

        order_1_classes = self.get_class_1stOrder(eval_data_changes)

        logging.info("Evaluating prediction accuracy")
        prediction_accuracies = []

        y_true_overall = []
        y_pred_overall = []

        for ind, k in enumerate(order_1_classes):
            x_train, y_true = self.get_train_test(eval_data_changes[ind, -self.weeks_to_eval - self.n_steps:],
                                                  eval_data_changes_volume[ind, -self.weeks_to_eval - self.n_steps:],
                                                  k[-self.weeks_to_eval - self.n_steps:])

            self.load_model(self.company_list[ind])

            y_pred = self.model.predict(x_train)
            y_pred = np.argmax(y_pred, axis=1)

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

    def do_pred(self, time_step=2):
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

            predict_date = last_date + datetime.timedelta(days=7 * (t + 1))

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

class Cnn3ClassWeekly(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = ["n_steps"]
        self.opt_args = ['ouput_dir', "days_to_eval", "model_dir", 'pred_dir', 'epochs']
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

        self.blank_model = None

    def do_init(self, args):
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"

        self.model_dir = args["model_dir"] if "model_dir" in args.keys() else self.output_dir+"training_dir/"+self.name+"/"
        self.eval_dir = args["eval_dir"] if "eval_dir" in args.keys() else self.output_dir+"eval_dir/"+self.name+"/"
        self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.model_file = os.path.join(os.path.dirname(self.model_dir), "model.npy")
        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")

        self.blank_filename = "blank.h5"
        self.blank_model = os.path.join(os.path.dirname(self.model_dir), self.blank_filename)

        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30
        self.weeks_to_eval = args["weeks_to_eval"] if "weeks_to_eval" in args.keys() else 4

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()
        self.company_list = self.db.get_list_companies()
        self.companies_count = self.db.get_companies_count()

        self.n_steps = args["n_steps"] if "n_steps" in args.keys() else 2
        self.n_classes = 3

        self.epochs = args["epochs"] if "epochs" in args.keys() else 50

        self.model = self.init_model()
        

    def init_model(self):

        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.n_steps, 1), padding="same"))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding="same"))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.save_model(model, "blank")

        return model

    def save_model(self, model, stock):
        file_name = stock+".h5"
        try:
            os.makedirs(self.model_dir)
        except OSError:
            logging.warning("Creation of the directory %s Falied (May already exist)" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        model.save_weights(os.path.join(self.model_dir, file_name))

        logging.info("Saving model %s", file_name)

    def load_model(self, stock):
        file_name = stock + ".h5"
        # with open(file_name, 'rb') as pkl:
        #     model = pickle.load(pkl)
        # return model
        logging.info("Loading model weights %s" % os.path.join(self.model_dir, file_name))
        self.model.load_weights(os.path.join(self.model_dir, file_name))

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


    def get_train_test(self, rows, classes):
        x_train = []
        y_train = []
        for k in range(len(rows)-self.n_steps):
            x_train += [rows[k:k+self.n_steps]]
            y_train += [classes[k+self.n_steps]]

        return np.expand_dims(np.array(x_train), axis=2), y_train

    def do_train(self):
        logging.info("Generating Train Data")
        train_data_changes = self.generate_train_data(column="high")

        logging.info("Generating Train Data: Done")

        classes_train = self.get_class_1stOrder(train_data_changes)

        logging.info("Training")

        for ind, k in enumerate(classes_train):
            self.load_model("blank")
            trimmed = np.trim_zeros(train_data_changes[ind])

            x_train, y_train  = self.get_train_test(trimmed, k[-len(trimmed):])
            y_train = to_categorical(y_train, num_classes=self.n_classes)

            metrics = self.model.fit(x_train, y_train, epochs=self.epochs, verbose=0)

            self.save_model(self.model, self.company_list[ind])


            logging.info("Training: Done, Accuracy: %s" % metrics.history['accuracy'][-1])

        # self.save_model(transition_matrix)
        return

    def do_eval(self):

        logging.info("Generating eval data")
        eval_data_changes = self.generate_eval_data(column="high")
        logging.info("Generating eval data: Done")

        order_1_classes = self.get_class_1stOrder(eval_data_changes)

        logging.info("Evaluating prediction accuracy")
        prediction_accuracies = []

        y_true_overall = []
        y_pred_overall = []

        for ind, k in enumerate(order_1_classes):

            x_train, y_true  = self.get_train_test(eval_data_changes[ind, -self.weeks_to_eval-self.n_steps:], k[-self.weeks_to_eval-self.n_steps:])

            self.load_model(self.company_list[ind])

            y_pred = self.model.predict(x_train)
            y_pred = np.argmax(y_pred, axis=1)

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
        with open(os.path.join(self.model_dir,file_name), 'rb') as pkl:
            logging.info("Loading model: %s" % os.path.join(self.model_dir, file_name))
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

        total_samples_avail = max_items -  self.weeks_to_eval

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
        train_data_changes, train_data_changes_a = self.generate_train_data(column="high")

        logging.info("Generating Train Data: Done")

        logging.info("Training")

        for k in range(train_data_changes_a.shape[0]):
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

        for ind, row in enumerate(eval_data_changes_a):
            row = np.trim_zeros(row)

            length_data = len(row)
            pred = []
            for k in range(self.weeks_to_eval):
                train_data = row[:(length_data-self.weeks_to_eval+k)]
                model = auto_arima(train_data, error_action='ignore', suppress_warnings=True, seasonal=True)
                model.fit(train_data)
                pred += [model.predict(n_periods=1)[0]]

            y_true = order_1_classes[ind][-self.weeks_to_eval:]

            y_pred = [a/b*c for a,b,c in zip(pred, eval_data_changes_a[ind,-self.weeks_to_eval:], eval_data_changes[ind, -self.weeks_to_eval:])]

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

class Markov3ClassWeeklyCRFVolume(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = ['order']
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

        self.n_classes = None
        self.n_classes_volume = None

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
        self.n_classes_volume = 3

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

    def generate_train_data(self, column="volume", stats="max"):
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
        volume_values = np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

        total_samples_avail = max_items - self.weeks_to_eval

        train_samples = values[:, :total_samples_avail]
        train_samples = (train_samples[:, 1:] - train_samples[:, 0:-1]) / train_samples[:, 0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample = volume_values[:, :total_samples_avail]
        volume_sample = (volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1]

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0

        return train_samples, volume_sample

    def generate_eval_data(self, column="volume", stats="max"):
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
        volume_values = np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

        total_samples_avail = max_items 

        train_samples = values[:, :total_samples_avail]
        train_samples = (train_samples[:, 1:] - train_samples[:, 0:-1]) / train_samples[:, 0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample = volume_values[:, :total_samples_avail]
        volume_sample = (volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1]

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0

        return train_samples, volume_sample

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

    def get_class_1stOrder_mean(self, mat):

        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)

        def get_log_values(data):
            data = np.log(data)
            data[np.isnan(data)] = 0
            data[np.isinf(data)] = 0

            return data

        for k in range(mat.shape[0]):
            row = np.trim_zeros(mat[k])
            temp = np.ones(mat[k].shape)
            mar_log = get_log_values(mat)
            row_log = get_log_values(row)

            temp[mar_log[k] > np.mean(row_log) + np.std(row_log)] = 2
            temp[mar_log[k] <= np.mean(row_log) - np.std(row_log)] = 0
            matclasses[k] = temp

        return matclasses


    def get_conditional_classes(self, classes_price, classes_volume):
        return (np.max(classes_price)+1)*classes_volume + classes_price

    def do_train(self):
        logging.info("Generating Train Data")
        train_data_changes, volume_change = self.generate_train_data(column="high")
        logging.info("Generating Train Data: Done")

        number_of_classes = self.n_classes**self.order*self.n_classes_volume
        order_1_classes = self.get_classes(train_data_changes)
        order_1_classes_volume = self.get_class_1stOrder_mean(volume_change)

        conditional_classes = self.get_conditional_classes(order_1_classes, order_1_classes_volume[:,self.order-1:])

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
        eval_price_changes, eval_volume_change = self.generate_eval_data(column="high")
        logging.info("Generating eval data: prices - Done")

        order_1_classes = self.get_classes(eval_price_changes)
        order_1_classes_volume = self.get_class_1stOrder_mean(eval_volume_change)

        conditional_classes = self.get_conditional_classes(order_1_classes, order_1_classes_volume[:,self.order-1:])

        logging.info("Evaluating prediction accuracy")
        prediction_accuracies = []

        class_multiplier = [k for k in reversed([(self.n_classes) ** k for k in range(self.order)])]

        y_true_overall = []
        y_pred_overall = []

        for ind, row in enumerate(conditional_classes):
            eval_data_for_this_row = row[-self.weeks_to_eval-1:]
            y_true = eval_data_for_this_row[1:]
            y_pred = []
            for k in eval_data_for_this_row[:-1]:
                initial_state = k
                y_pred += [self.predict_next_state(initial_state, transision_matrix[ind])]

            y_true_order_1 = [r % self.n_classes_volume for r in y_true]
            y_pred_order_1 = [r % self.n_classes_volume for r in y_pred]

            for k in class_multiplier[:-1]:
                y_true_order_1 = [r%k for r in y_true_order_1]
                y_pred_order_1 = [r%k for r in y_pred_order_1]

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

class GlobalMarkov3ClassWeeklyCRFVolume(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = ['order']
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

        self.n_classes = None
        self.n_classes_volume = None

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
        self.n_classes_volume = 3

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

    def generate_train_data(self, column="volume", stats="max"):
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
        volume_values = np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

        total_samples_avail = max_items - self.weeks_to_eval

        train_samples = values[:, :total_samples_avail]
        train_samples = (train_samples[:, 1:] - train_samples[:, 0:-1]) / train_samples[:, 0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample = volume_values[:, :total_samples_avail]
        volume_sample = (volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1]

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0

        return train_samples, volume_sample

    def generate_eval_data(self, column="volume", stats="max"):
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
        volume_values = np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

        total_samples_avail = max_items

        train_samples = values[:, :total_samples_avail]
        train_samples = (train_samples[:, 1:] - train_samples[:, 0:-1]) / train_samples[:, 0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample = volume_values[:, :total_samples_avail]
        volume_sample = (volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1]

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0

        return train_samples, volume_sample

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

    def get_class_1stOrder_mean(self, mat):

        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)

        def get_log_values(data):
            data = np.log(data)
            data[np.isnan(data)] = 0
            data[np.isinf(data)] = 0

            return data

        for k in range(mat.shape[0]):
            row = np.trim_zeros(mat[k])
            temp = np.ones(mat[k].shape)
            mar_log = get_log_values(mat)
            row_log = get_log_values(row)

            temp[mar_log[k] > np.mean(row_log) + np.std(row_log)] = 2
            temp[mar_log[k] <= np.mean(row_log) - np.std(row_log)] = 0
            matclasses[k] = temp

        return matclasses


    def get_conditional_classes(self, classes_price, classes_volume):
        return (np.max(classes_price)+1)*classes_volume + classes_price

    def do_train(self):
        logging.info("Generating Train Data")
        train_data_changes, volume_change = self.generate_train_data(column="high")
        logging.info("Generating Train Data: Done")

        number_of_classes = self.n_classes**self.order*self.n_classes_volume
        order_1_classes = self.get_classes(train_data_changes)
        order_1_classes_volume = self.get_class_1stOrder_mean(volume_change)

        conditional_classes = self.get_conditional_classes(order_1_classes, order_1_classes_volume[:,self.order-1:])

        transition_matrix = np.zeros((number_of_classes, number_of_classes))

        logging.info("Training")

        for k in range(train_data_changes.shape[0]):
            changes = np.trim_zeros(train_data_changes[k])
            tra_2 = conditional_classes[k, -len(changes):]
            for tminus1, t in zip(tra_2[:-1], tra_2[1:]):
                transition_matrix[tminus1, t] += 1

        logging.info("Training: Done")

        self.save_model(transition_matrix)
        return

    def do_eval(self):
        logging.info("Loading transition matrix")
        transision_matrix = self.load_model()

        logging.info("Generating eval data: prices")
        eval_price_changes, eval_volume_change = self.generate_eval_data(column="high")
        logging.info("Generating eval data: prices - Done")

        order_1_classes = self.get_classes(eval_price_changes)
        order_1_classes_volume = self.get_class_1stOrder_mean(eval_volume_change)

        conditional_classes = self.get_conditional_classes(order_1_classes, order_1_classes_volume[:,self.order-1:])

        logging.info("Evaluating prediction accuracy")
        prediction_accuracies = []

        class_multiplier = [k for k in reversed([(self.n_classes) ** k for k in range(self.order)])]

        y_true_overall = []
        y_pred_overall = []

        for ind, row in enumerate(conditional_classes):
            eval_data_for_this_row = row[-self.weeks_to_eval-1:]
            y_true = eval_data_for_this_row[1:]
            y_pred = []
            for k in eval_data_for_this_row[:-1]:
                initial_state = k
                y_pred += [self.predict_next_state(initial_state, transision_matrix)]

            y_true_order_1 = [r % self.n_classes_volume for r in y_true]
            y_pred_order_1 = [r % self.n_classes_volume for r in y_pred]

            for k in class_multiplier[:-1]:
                y_true_order_1 = [r%k for r in y_true_order_1]
                y_pred_order_1 = [r%k for r in y_pred_order_1]

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

class GlobalMarkov3ClassWeekly(FEModel):

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

        transition_matrix = np.zeros((self.n_classes**self.order, self.n_classes**self.order))


        logging.info("Training")

        for k in range(train_data_changes.shape[0]):
            changes = np.trim_zeros(train_data_changes[k])
            tra = classes[k, -len(changes):]

            for tminus1, t in zip(tra[:-1], tra[1:]):
                transition_matrix[tminus1, t] += 1

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
                y_pred += [self.predict_next_state(initial_state, transision_matrix)]

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

class Markov2ClassWeeklyCRFVolume(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = ['order']
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

        self.n_classes = None
        self.n_classes_volume = None

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

        self.n_classes = 2
        self.n_classes_volume = 3

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

    def generate_train_data(self, column="volume", stats="max"):
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
        volume_values = np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

        total_samples_avail = max_items - self.weeks_to_eval

        train_samples = values[:, :total_samples_avail]
        train_samples = (train_samples[:, 1:] - train_samples[:, 0:-1]) / train_samples[:, 0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample = volume_values[:, :total_samples_avail]
        volume_sample = (volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1]

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0

        return train_samples, volume_sample

    def generate_eval_data(self, column="volume", stats="max"):
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
        volume_values = np.zeros((total_companies, max_items))

        for i in range(len(company_list)):
            logging.info("Organizing Data: Processing - %s/%s" % (i, len(company_list)))
            values_fetch = values_array[i]
            values[i, max_items - len(values_fetch):max_items] = values_fetch

            volume_fetch = volume_array[i]
            volume_values[i, max_items - len(values_fetch):max_items] = volume_fetch

        total_samples_avail = max_items

        train_samples = values[:, :total_samples_avail]
        train_samples = (train_samples[:, 1:] - train_samples[:, 0:-1]) / train_samples[:, 0:-1]

        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        volume_sample = volume_values[:, :total_samples_avail]
        volume_sample = (volume_sample[:, 1:] - volume_sample[:, 0:-1]) / volume_sample[:, 0:-1]

        volume_sample[np.isnan(volume_sample)] = 0
        volume_sample[np.isinf(volume_sample)] = 0

        return train_samples, volume_sample

    def get_classes(self, data, labels={0: 0}):
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

    def get_class_1stOrder_mean(self, mat):

        mat = np.array(mat)
        matclasses = np.zeros(mat.shape, dtype=np.int)

        def get_log_values(data):
            data = np.log(data)
            data[np.isnan(data)] = 0
            data[np.isinf(data)] = 0

            return data

        for k in range(mat.shape[0]):
            row = np.trim_zeros(mat[k])
            temp = np.ones(mat[k].shape)
            mar_log = get_log_values(mat)
            row_log = get_log_values(row)

            temp[mar_log[k] > np.mean(row_log) + np.std(row_log)] = 2
            temp[mar_log[k] <= np.mean(row_log) - np.std(row_log)] = 0
            matclasses[k] = temp

        return matclasses

    def get_conditional_classes(self, classes_price, classes_volume):
        return (np.max(classes_price)+1)*classes_volume + classes_price

    def do_train(self):
        logging.info("Generating Train Data")
        train_data_changes, volume_change = self.generate_train_data(column="high")
        logging.info("Generating Train Data: Done")

        number_of_classes = self.n_classes**self.order*self.n_classes_volume
        order_1_classes = self.get_classes(train_data_changes)
        order_1_classes_volume = self.get_class_1stOrder_mean(volume_change)

        conditional_classes = self.get_conditional_classes(order_1_classes, order_1_classes_volume[:,self.order-1:])

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
        eval_price_changes, eval_volume_change = self.generate_eval_data(column="high")
        logging.info("Generating eval data: prices - Done")

        order_1_classes = self.get_classes(eval_price_changes)
        order_1_classes_volume = self.get_class_1stOrder_mean(eval_volume_change)

        conditional_classes = self.get_conditional_classes(order_1_classes, order_1_classes_volume[:,self.order-1:])

        logging.info("Evaluating prediction accuracy")
        prediction_accuracies = []

        class_multiplier = [k for k in reversed([(self.n_classes) ** k for k in range(self.order)])]

        y_true_overall = []
        y_pred_overall = []

        for ind, row in enumerate(conditional_classes):
            eval_data_for_this_row = row[-self.weeks_to_eval-1:]
            y_true = eval_data_for_this_row[1:]
            y_pred = []
            for k in eval_data_for_this_row[:-1]:
                initial_state = k
                y_pred += [self.predict_next_state(initial_state, transision_matrix[ind])]

            y_true_order_1 = [r % self.n_classes_volume for r in y_true]
            y_pred_order_1 = [r % self.n_classes_volume for r in y_pred]

            if class_multiplier[:-1]:
                for k in class_multiplier[:-1]:
                    y_true_order_1 = [r%k for r in y_true_order_1]
                    y_pred_order_1 = [r%k for r in y_pred_order_1]
            else:
                y_true_order_1 = [r//2 for r in y_true_order_1]
                y_pred_order_1 = [r//2 for r in y_pred_order_1]

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
        logging.info("Risk: %s" % (confusion[0, 1]))

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

class Markov2ClassWeekly(FEModel):

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

        self.n_classes = 2

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

    def get_classes(self, data, labels={0: 0}):
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
        logging.info("Risk: %s" % (confusion[0,1]))


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