from FE_Models.model_DB_Reader import DB_Ops
import logging, os, pickle
import numpy as np

logging = logging.getLogger("main")


class FEMetrics:
    def __init__(self, name, req_args, opt_args):
        self.name = name
        self.req_args = req_args
        self.opt_args = opt_args

    def get_args(self):
        return {"required": self.req_args, "optional": self.opt_args}

    def do_compute(self, args):
        pass

    def do_eval(self):
        pass


class HubberRegressionDaily(FEMetrics):
    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = []
        self.opt_args = ['ouput_dir', "days_to_eval", "model_dir", 'metric_dir' ]
        FEMetrics.__init__(self, self.name, self.req_args, self.opt_args)

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

        self.name = self.name
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"

        self.model_dir = args["model_dir"] if "model_dir" in args.keys() else self.output_dir+"training_dir/"+self.name+"/"
        self.eval_dir = args["eval_dir"] if "eval_dir" in args.keys() else self.output_dir+"eval_dir/"+self.name+"/"
        # self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"
        self.metric_dir = args[
            "metric_dir"] if "metric_dir" in args.keys() else self.output_dir + "metric_dir/" + self.name + "/"

        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.metric_file = os.path.join(os.path.dirname(self.metric_dir), "metric.json")

        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30
        self.weeks_to_eval = args["weeks_to_eval"] if "weeks_to_eval" in args.keys() else 8

        logging.info("Test")
        logging.info("Initializing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        self.companies_count = self.db.get_companies_count()
        self.company_list = self.db.get_list_companies()

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

        return train_samples, train_samples_actual

    def generate_eval_data(self, column = "volume"):
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

        total_samples_avail  = max_items

        dates_samples = dates[:,:total_samples_avail]
        train_samples_actual = values[:,:total_samples_avail]

        train_samples = (train_samples_actual[:,1:] - train_samples_actual[:,0:-1])/train_samples_actual[:,0:-1]

        #remove zeros, and infs
        train_samples[np.isnan(train_samples)] = 0
        train_samples[np.isinf(train_samples)] = 0

        return train_samples, train_samples_actual, dates_samples

    def save_model(self, transition_matrix):
        try:
            os.makedirs(self.metric_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        np.save(self.metric_file, transition_matrix)
        logging.info("Saving metrics to %s", self.metric_file)

    def load_model(self):
        return np.load(self.model_file)

    def do_compute(self):

        logging.info("Generating Compute Data")
        train_data_changes, train_data_actual = self.generate_train_data(column="high")

        from sklearn.linear_model import HuberRegressor

        reg_data = []

        for ind, k in enumerate(train_data_actual):
            # logging.info("Computing for %s " % ind)
            k = np.trim_zeros(k)
            model = HuberRegressor()
            x = np.arange((len(k))).reshape(-1,1)
            y = np.array(k).reshape(-1,1)
            # model.fit(x, y)

            if y.shape[0] > 100:
                try:
                    model.fit(x, y)

                    reg_data += [[model.coef_[0], model.scale_, model.intercept_]]
                except:
                    logging.info("--------------- %s Unable to fit -------------" % ind)
                    reg_data += [[-100, 0, 0]]

            else:
                logging.info("--------------- %s unreliable, < 100 days of data available -------------" % ind)
                reg_data += [[-100, 0, 0]]

            logging.info("Prameters for %s = %s" % (ind, reg_data[-1]))

        logging.info("Training: Done")

        self.save_metric_output(reg_data)
        return

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

    def save_metric_output(self, data):
        try:
            os.makedirs(self.metric_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.metric_dir)
        else:
            logging.info("Successfully created the directory %s " % self.metric_dir)

        logging.info("Writing metric to %s", self.eval_file)

        with open(self.metric_file, 'wb') as outfile:
            pickle.dump(data, outfile)
        outfile.close()

    def do_eval(self):
        changes_all, actual_all, dates = self.generate_eval_data("high")

        from sklearn.linear_model import HuberRegressor
        eval_data = dict()

        for day in range(self.days_to_eval):
            logging.info("Computing for day %s" % (day))
            days_behind = self.days_to_eval - day

            dates_actual = dates[:, :-days_behind]
            data_actual = actual_all[:, :-days_behind]
            # data_change = changes_all[:, :-days_behind]

            reg_data, dates_of_pred = [], []

            for ind, k in enumerate(data_actual):
                dates_of_pred += [dates_actual[ind, -1]]
                k = np.trim_zeros(k)

                model = HuberRegressor()

                x = np.arange((len(k))).reshape(-1, 1)
                y = np.array(k).reshape(-1, 1)

                if y.shape[0] > 100:
                    model.fit(x, y)

                    reg_data += [[model.coef_[0], model.scale_, model.intercept_]]

                else:
                    reg_data += [[-100, 0, 0]]

            reg_data = np.array(reg_data)
            dates_of_pred = np.array(dates_of_pred)

            eval_data[day] = {"metric": reg_data, "dates": dates_of_pred}

        self.save_eval_output(eval_data)

