from FE_Models.model_DB_Reader import DB_Ops
import pickle, datetime, logging, os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

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

        for ind, row in enumerate(order_2_classes):
            eval_data_for_this_row = row[-self.weeks_to_eval-1:]
            y_true = eval_data_for_this_row[1:]
            y_pred = []
            for k in eval_data_for_this_row[:-1]:
                initial_state = k
                y_pred += [self.predict_next_state(initial_state, transision_matrix[ind])]
            # prediction_accuracies_2 += [accuracy_score(y_true, y_pred)]
            # logging.info("%s, 2nd Order, GT: %s, Pred: %s, acc: %s" % (ind, y_true, y_pred, prediction_accuracies_2[-1]))

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
