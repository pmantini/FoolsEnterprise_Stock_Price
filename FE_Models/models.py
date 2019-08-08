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


class markov1(FEModel):

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

        max_items = self.db.get_max_rows() - 1

        values = np.zeros((total_companies, max_items))

        i = 0
        for k in company_list:
            if i == 0:
                _, dates_fetch = self.db.get_values_company(company_sym=k)
            values_fetch, _ = self.db.get_values_company(company_sym=k, columns=column)
            values_fetch = values_fetch[:-1]
            values[i, max_items - len(values_fetch):max_items] = values_fetch
            i += 1

        dates = dates_fetch[-self.days_to_eval:]

        eval_samples = values[:, -self.days_to_eval:]

        eval_samples = (eval_samples[:, 1:] - eval_samples[:, 0:-1]) / eval_samples[:, 0:-1]

        # remove zeros, and infs
        eval_samples[np.isnan(eval_samples)] = 0
        eval_samples[np.isinf(eval_samples)] = 0

        return eval_samples, dates

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

        pred_samples = values[:, -2:]

        pred_samples = (pred_samples[:, 1:] - pred_samples[:, 0:-1]) / pred_samples[:, 0:-1]

        # remove zeros, and infs
        pred_samples[np.isnan(pred_samples)] = 0
        pred_samples[np.isinf(pred_samples)] = 0

        # print(pred_samples)

        return pred_samples, dates



    def get_class(self, mat, labels={0: -0.05, 1: -0.025, 2: 0, 3: 0.025, 4: 0.05}):
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
        train_data = self.generate_train_data(column="open")

        number_of_classes = 6


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

        eval_data, dates = self.generate_eval_data()

        eval_classes = self.get_class(eval_data)
        eval_pred = np.zeros(eval_classes.shape, dtype=np.int)


        i = 0
        for eval_element in eval_data:
            j = 0
            for state in eval_classes[i]:
                initial_state = state
                if j == 0:
                    eval_pred[i, j] = initial_state

                this_trans = transision_matrix[i]

                this_trans /= this_trans.sum(axis=1).reshape(-1, 1)

                trans_cumsum = np.cumsum(this_trans, axis=1)

                randomvalue = np.random.random()

                class_iter = 0
                for pty in trans_cumsum[initial_state]:
                    if randomvalue < pty:
                        eval_pred[i, j] = class_iter
                        break;
                    class_iter += 1
                j += 1
            i += 1

        output = dict()
        from sklearn.metrics import accuracy_score

        eval_iter = 0
        all_returns = []
        for y_true, y_pred in zip(eval_classes, eval_pred):

            output[eval_iter] = {"accuracy": accuracy_score(y_true, y_pred)}

            returns, returns2, returns3, returns4 = [], [], [], []

            for class_th in [2, 3, 4]:
                best_gusses = y_pred > class_th
                for eval_value, guesses in zip(eval_data[eval_iter], best_gusses):
                    if guesses:
                        returns += [eval_value]
                        all_returns += [eval_value]

                if returns:
                    output[eval_iter]["returns"+str(class_th)] = np.mean(returns)
                else:
                    output[eval_iter]["returns" + str(class_th)] = 0

            eval_iter += 1

        y_true = eval_classes.flatten()
        y_pred = eval_pred.flatten()


        output["overall"] = {"accuracy": accuracy_score(y_true, y_pred)}
        # print(output)

        for k in output:
            print(output[k])

        self.save_eval_output(output)

    def do_pred(self):
        #### Computes overall average accuracy, per stock accuracy
        # evaluation
        transision_matrix = self.load_model()

        pred_data, dates = self.generate_pred_data()

        pred_classes = self.get_class(pred_data)
        # pred_pred = np.zeros(pred_classes.shape, dtype=np.int)
        pred_pred = dict()

        i = 0
        last_date = datetime.datetime.strptime(dates[1], "%Y-%m-%d")
        predict_date = last_date + datetime.timedelta(days=1)
        for pred_element in pred_classes:
            pred_pred[i] = {"date": str(predict_date).split(" ")[0]}
            initial_state = pred_element[0]
            this_trans = transision_matrix[i]
            this_trans /= this_trans.sum(axis=1).reshape(-1, 1)
            trans_cumsum = np.cumsum(this_trans, axis=1)
            randomvalue = np.random.random()
            class_iter = 0

            for pty in trans_cumsum[initial_state]:
                if randomvalue < pty:
                    pred_pred[i]["prediction"] = class_iter
                    # pred_pred[i] = class_iter
                    break
                class_iter += 1
            i += 1

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



class points(FEModel):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = []
        self.opt_args = ['ouput_dir', 'days_per_sample', 'batch_size', 'epochs', 'epoch_per_batch', 'training_exclude_days', "tensorboard_delete", "tensorboard_dir", "model_dir", "days_to_eval", 'pred_dir']
        FEModel.__init__(self, self.name, self.req_args, self.opt_args)

        self.company_list = None
        self.black_list = None
        self.db_folder = None
        self.output_dir = None

        self.days_per_sample = None
        self.batch_size = None
        self.epochs = None
        self.epochs_per_batch = None

        self.db = None

        self.input_shape = None
        self.points_model = None

        self.model_dir = None

        self.model_file = None
        self.model_weights_file = None


    def do_init(self, args):
        self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"

        self.days_per_sample = int(args["days_per_sample"]) if "days_per_sample" in args.keys() else 200
        self.batch_size = args["batch_size"] if "batch_size" in args.keys() else 30
        self.epochs = int(args["epochs"]) if "epochs" in args.keys() else 10
        self.epochs_per_batch = args["epochs_per_batch"] if "epochs_per_batch" in args.keys() else 10

        self.training_exclude_days = args["training_exclude_days"] if "training_exclude_days" in args.keys() else 30
        self.tensorboard_dir = args["tensorboard_dir"] if "tensorboard_dir" in args.keys() else self.output_dir+"tensorboard_logs/"+self.name
        self.tensorboard_delete = args["tensorboard_delete"] if "tensorboard_delete" in args.keys() else True
        self.model_dir = args["model_dir"] if "model_dir" in args.keys() else self.output_dir+"training_dir/"+self.name+"/"
        self.eval_dir = args["eval_dir"] if "eval_dir" in args.keys() else self.output_dir+"eval_dir/"+self.name+"/"
        self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"

        self.model_file = os.path.join(os.path.dirname(self.model_dir), "model.json")
        self.model_weights_file = os.path.join(os.path.dirname(self.model_dir), "model.h5")
        self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
        self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")


        #Evaluation Params
        self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30

        logging.info("Test")
        logging.info("Initializeing, with params: %s", str([k for k in args.items()]))

        self.db = DB_Ops()

        companies_count = self.db.get_companies_count()
        self.input_shape = (companies_count, self.days_per_sample, 1)

        self.points_model = self.model_init(self.input_shape)
        self.tensorboard = TensorBoard(log_dir=self.tensorboard_dir, update_freq=10000)

    def model_init(self, input_shape):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Conv2D(512, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        # model.add(Dense(1000, activation='relu'))

        model.add(Dense(491, activation='relu'))

        model.add(Dense(1, activation = 'sigmoid'))


        logging.info(model.to_json())
        model.summary()

        return model


    def compile_model(self, model):
        # optimizer = Adam(0.0002, 0.5)
        # optimizer = Adam()
        # optimizer = SGD()
        optimizer = Nadam(lr=0.0002)
        logging.info("Compiling...")
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        return model

    def save_model(self, model):
        #Create Folder
        try:
            os.makedirs(self.model_dir)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.model_dir)
        else:
            logging.info("Successfully created the directory %s " % self.model_dir)

        model_json = model.to_json()

        logging.info("Saved model to %s", self.model_file)
        with open(self.model_file, "w") as json_file:
            json_file.write(model_json)

        model.save_weights(self.model_weights_file)
        logging.info("Saved model weights to %s", self.model_weights_file)


    def load_model(self):
        print(self.model_file, self.model_weights_file)
        # load json and create model
        json_file = open(self.model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(self.model_weights_file)
        logging.info("Loaded model %s, with weights %s", self.model_file, self.model_weights_file)

        return loaded_model


    def generate_train_eval_data(self, exclude_rows_from_end = 30, column = "open"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = self.db.get_max_rows() - exclude_rows_from_end

        values = np.zeros((total_companies, max_items))

        i = 0
        for k in company_list:
            values_fetch, _ = self.db.get_values_company(company_sym=k, columns=column)
            values_fetch = values_fetch[:-exclude_rows_from_end]
            values[i, max_items - len(values_fetch):max_items] = values_fetch
            i += 1

        total_samples_avail  = max_items - self.days_per_sample
        # train_sample_size = int(train_partiion*total_samples_avail)
        #
        # random_samples = random.sample(range(0, total_samples_avail), total_samples_avail)

        random_samples = [k for k in range(total_samples_avail)]

        train_samples = random_samples[:total_samples_avail-self.batch_size]
        eval_samples = random_samples[self.batch_size:]

        if len(eval_samples) < self.days_per_sample:
            logging.error("Not enough samples to create valuation set, only %s available, require %s: Exiting", len(eval_samples), self.days_per_sample)
            exit()

        train_iter = 0
        eval_iter = 0

        epoch_iter = 0
        batch_count = 0

        x_train, y_train = None, None
        x_eval, y_eval = None, None
        while True:

            if epoch_iter >= self.epochs:
                logging.info("Max epochs reached: Exiting")
                raise StopIteration

            if train_iter >= len(train_samples):
                epoch_iter += 1
                train_iter = 0
                eval_iter = 0

            eval_iter = train_iter if eval_iter >= len(eval_samples) else eval_iter

            if x_train is None:

                temp_sample = values[:, train_samples[train_iter]:train_samples[train_iter] + self.days_per_sample + 1]

                x_train = temp_sample[:, :-1].reshape((self.days_per_sample) * total_companies)
                y_train = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
                y_train[np.isnan(y_train)] = 0
                y_train[np.isinf(y_train)] = 0
                y_train = 1 if np.mean(y_train) > 0 else 0

                temp_sample = values[:, eval_samples[eval_iter]:eval_samples[eval_iter] + self.days_per_sample + 1]

                x_eval = temp_sample[:, :-1].reshape((self.days_per_sample) * total_companies)
                y_eval = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
                y_eval[np.isnan(y_eval)] = 0
                y_eval[np.isinf(y_eval)] = 0
                y_eval = 1 if np.mean(y_eval) > 0 else 0

                train_iter += 1
                eval_iter += 1
                batch_count += 1

            else:
                if batch_count >= self.batch_size:
                    x_train = x_train.reshape(self.batch_size, total_companies, self.days_per_sample)
                    y_train = y_train.reshape(self.batch_size, 1)

                    x_eval = x_eval.reshape(self.batch_size, total_companies, self.days_per_sample)
                    y_eval = y_eval.reshape(self.batch_size, 1)

                    yield x_train, y_train, x_eval, y_eval
                    x_train, y_train, x_eval, y_eval = None, None, None, None
                    batch_count = 0
                    train_iter += 1
                    eval_iter += 1

                    continue


                temp_sample = values[:, train_samples[train_iter]:train_samples[train_iter] + self.days_per_sample + 1]
                temp_samplex = temp_sample[:, :-1].reshape((self.days_per_sample) * total_companies)
                x_train = np.vstack((x_train, temp_samplex))


                temp_sampley = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]

                temp_sampley[np.isnan(temp_sampley)] = 0
                temp_sampley[np.isinf(temp_sampley)] = 0
                temp_sampley = 1 if np.mean(temp_sampley) > 0 else 0
                y_train = np.append(y_train, temp_sampley)

                temp_sample = values[:, eval_samples[eval_iter]:eval_samples[eval_iter] + self.days_per_sample + 1]
                temp_samplex = temp_sample[:, :-1].reshape((self.days_per_sample) * total_companies)
                x_eval = np.vstack((x_eval, temp_samplex))

                temp_sampley = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
                temp_sampley[np.isnan(temp_sampley)] = 0
                temp_sampley[np.isinf(temp_sampley)] = 0
                temp_sampley = 1 if np.mean(temp_sampley) > 0 else 0
                y_eval = np.append(y_eval, temp_sampley)

                batch_count += 1
                train_iter += 1
                eval_iter += 1


    def generate_eval_data(self, column = "open"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = self.db.get_max_rows() - 1

        values = np.zeros((total_companies, max_items))

        i = 0
        for k in company_list:
            if i == 0:
                _, dates_fetch = self.db.get_values_company(company_sym=k)
            values_fetch, _ = self.db.get_values_company(company_sym=k, columns=column)
            values_fetch = values_fetch[:-1]
            values[i, max_items - len(values_fetch):max_items] = values_fetch
            i += 1

        dates = dates_fetch[-self.days_to_eval:]

        iter = max_items - self.days_per_sample - self.days_to_eval - 1

        x_train, y_train, Dates = None, None, None
        i = 0
        while True:

            if x_train is None:

                temp_sample = values[:, iter:iter + self.days_per_sample + 1]

                x_train = temp_sample[:, :-1].reshape((self.days_per_sample) * total_companies)

                y_train = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
                y_train[np.isnan(y_train)] = 0
                y_train[np.isinf(y_train)] = 0
                y_train = 1 if np.mean(y_train) > 0 else 0

                iter += 1

            else:
                if iter + self.days_per_sample + 1 >= max_items:
                    x_train = x_train.reshape(self.days_to_eval, total_companies, self.days_per_sample)
                    y_train = y_train.reshape(self.days_to_eval, 1)

                    return x_train, y_train, dates

                temp_sample = values[:, iter:iter + self.days_per_sample + 1]
                temp_samplex = temp_sample[:, :-1].reshape((self.days_per_sample) * total_companies)
                x_train = np.vstack((x_train, temp_samplex))

                temp_sampley = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
                temp_sampley[np.isnan(temp_sampley)] = 0
                temp_sampley[np.isinf(temp_sampley)] = 0
                temp_sampley = 1 if np.mean(temp_sampley) > 0 else 0
                y_train = np.append(y_train, temp_sampley)

                iter += 1


    def generate_pred_data(self, column = "open"):
        company_list = self.db.get_list_companies()
        total_companies = self.db.get_companies_count()

        max_items = self.db.get_max_rows()

        values = np.zeros((total_companies, max_items))

        i = 0
        for k in company_list:
            values_fetch , _ = self.db.get_values_company(company_sym=k, columns=column)
            values[i, max_items - len(values_fetch):max_items] = values_fetch
            i += 1

        x_train = values[:, -self.days_per_sample:]
        x_train.reshape(1, total_companies, self.days_per_sample)

        return x_train

    def do_train(self):
        if self.tensorboard_delete:
            logging.info("FLAG tensorboard_delete is True, Deleting existing files")
            shutil.rmtree(self.tensorboard_dir)


        self.points_model = self.compile_model(self.points_model)

        #setting training data to everything except last n days
        data_vol = self.generate_train_eval_data(exclude_rows_from_end=self.training_exclude_days, column="volume")
        data = self.generate_train_eval_data(exclude_rows_from_end=self.training_exclude_days)

        while True:
            try:
                x_t, _, x_e, _ = next(data_vol)
                _, y_t, _, y_e = next(data)

                i = 0
                for k in x_t:
                    j = 0
                    for coulmn in k:
                        max_value = np.max(coulmn)
                        if max_value:
                            x_t[i, j] = x_t[i, j] / max_value
                        j += 1
                    i += 1

                x_t = x_t.reshape(self.batch_size, self.db.get_companies_count(), self.days_per_sample, 1)
                y_t = y_t.reshape(self.batch_size, -1)

                i = 0
                for k in x_e:
                    j = 0
                    for coulmn in k:
                        max_value = np.max(coulmn)
                        if max_value:
                            x_e[i, j] = x_e[i, j] / max_value
                        j += 1
                    i += 1

                x_e = x_e.reshape(self.batch_size, self.db.get_companies_count(), self.days_per_sample, 1)
                y_e = y_e.reshape(self.batch_size, -1)

                self.points_model.fit(x_t, y_t, batch_size=self.batch_size, epochs = self.epochs_per_batch, validation_data=(x_e, y_e), callbacks=[self.tensorboard])
            except StopIteration:
                #Save Model
                logging.info("End Training!")
                self.save_model(self.points_model)
                break



    def do_eval(self):
        self.points_model = self.load_model()
        self.compile_model(self.points_model)

        x_, _, d_ = self.generate_eval_data(column="volume")
        _, y_, _ = self.generate_eval_data()
        i = 0
        for k in x_:
            j = 0
            for coulmn in k:
                max_value = np.max(coulmn)
                if max_value:
                    x_[i, j] = x_[i, j] / max_value
                j += 1
            i += 1

        x_ = x_.reshape(self.days_to_eval, self.db.get_companies_count(), self.days_per_sample, 1)
        y_ = y_.reshape(self.days_to_eval, -1)

        predictions = self.points_model.predict(x_)


        output = dict()
        output['dates'] = d_
        output['prediction'] = list(predictions.reshape(self.days_to_eval))
        output['ground_truth'] = list(y_.reshape(self.days_to_eval))

        print(output)

        self.save_eval_output(output)

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

        logging.info("Writing evaluation output to %s", self.pred_dir)

        with open(self.pred_file, 'wb') as outfile:
            pickle.dump(data, outfile)

        outfile.close()


    def do_pred(self):
        self.points_model = self.load_model()
        self.compile_model(self.points_model)

        x_ = self.generate_pred_data(column="volume")

        i = 0
        for k in x_:
            j = 0
            for coulmn in k:
                max_value = np.max(coulmn)
                if max_value:
                    x_[i, j] = x_[i, j] / max_value
                j += 1
            i += 1

        x_ = x_.reshape(1, self.db.get_companies_count(), self.days_per_sample, 1)

        prediction = self.points_model.predict(x_)


        output = dict()
        output['prediction'] = list(prediction.reshape(1))
        print(output)
        self.save_pred_output(output)


# class ranks(FEModel):
#
#     def __init__(self, args):
#         self.name = self.__class__.__name__
#         self.req_args = []
#         self.opt_args = ['ouput_dir', 'days_per_sample', 'batch_size', 'epochs', 'epoch_per_batch', 'training_exclude_days', "tensorboard_delete", "tensorboard_dir", "model_dir", "days_to_eval", 'pred_dir']
#         FEModel.__init__(self, self.name, self.req_args, self.opt_args)
#
#         self.company_list = None
#         self.black_list = None
#         self.db_folder = None
#         self.output_dir = None
#
#         self.days_per_sample = None
#         self.batch_size = None
#         self.epochs = None
#         self.epochs_per_batch = None
#
#         self.db = None
#
#         self.input_shape = None
#         self.points_model = None
#
#         self.model_dir = None
#
#         self.model_file = None
#         self.model_weights_file = None
#
#
#     def do_init(self, args):
#         self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"
#
#         self.days_per_sample = int(args["days_per_sample"]) if "days_per_sample" in args.keys() else 200
#         self.batch_size = args["batch_size"] if "batch_size" in args.keys() else 30
#         self.epochs = int(args["epochs"]) if "epochs" in args.keys() else 10
#         self.epochs_per_batch = args["epochs_per_batch"] if "epochs_per_batch" in args.keys() else 10
#
#         self.training_exclude_days = args["training_exclude_days"] if "training_exclude_days" in args.keys() else 30
#         self.tensorboard_dir = args["tensorboard_dir"] if "tensorboard_dir" in args.keys() else self.output_dir+"tensorboard_logs/"+self.name
#         self.tensorboard_delete = args["tensorboard_delete"] if "tensorboard_delete" in args.keys() else True
#         self.model_dir = args["model_dir"] if "model_dir" in args.keys() else self.output_dir+"training_dir/"+self.name+"/"
#         self.eval_dir = args["eval_dir"] if "eval_dir" in args.keys() else self.output_dir+"eval_dir/"+self.name+"/"
#         self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"
#
#         self.model_file = os.path.join(os.path.dirname(self.model_dir), "model.json")
#         self.model_weights_file = os.path.join(os.path.dirname(self.model_dir), "model.h5")
#         self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
#         self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")
#
#
#         #Evaluation Params
#         self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30
#
#         logging.info("Test")
#         logging.info("Initializeing, with params: %s", str([k for k in args.items()]))
#
#         self.db = DB_Ops()
#
#         self.companies_count = self.db.get_companies_count()
#         self.input_shape = (self.companies_count, self.days_per_sample-2, 1)
#
#         self.points_model = self.model_init(self.input_shape)
#         self.tensorboard = TensorBoard(log_dir=self.tensorboard_dir, update_freq=10000)
#
#     def model_init(self, input_shape):
#         # model = Sequential()
#         #
#         # model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
#         #                  activation='relu',
#         #                  input_shape=input_shape))
#         # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#         # model.add(Conv2D(64, (3, 3), activation='relu'))
#         # model.add(MaxPooling2D(pool_size=(2, 2)))
#         # model.add(Conv2D(128, (3, 3), activation='relu'))
#         # model.add(MaxPooling2D(pool_size=(2, 2)))
#         # model.add(Conv2D(256, (3, 3), activation='relu'))
#         # model.add(MaxPooling2D(pool_size=(2, 2)))
#         # model.add(Flatten())
#         # model.add(Dense(1000, activation='relu'))
#         # model.add(Dense(750, activation='relu'))
#         # model.add(Dense(self.companies_count, activation='sigmoid'))
#
#         model = Sequential()
#         model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
#                          input_shape=input_shape))
#         model.add(Conv2D(32, (3, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.25))
#
#         model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#         model.add(Conv2D(64, (3, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.25))
#
#         model.add(Flatten())
#         model.add(Dense(128, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(self.companies_count, activation='sigmoid'))
#
#         logging.info(model.to_json())
#         model.summary()
#
#         return model
#
#
#     def compile_model(self, model):
#
#         logging.info("Compiling...")
#         # rms = optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
#         # sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#         model.compile(loss='binary_crossentropy',
#                       optimizer='adam',
#                       metrics=['mae'])
#
#         return model
#
#     def save_model(self, model):
#         #Create Folder
#         try:
#             os.makedirs(self.model_dir)
#         except OSError:
#             logging.warning("Creation of the directory %s failed" % self.model_dir)
#         else:
#             logging.info("Successfully created the directory %s " % self.model_dir)
#
#         model_json = model.to_json()
#
#         logging.info("Saved model to %s", self.model_file)
#         with open(self.model_file, "w") as json_file:
#             json_file.write(model_json)
#
#         model.save_weights(self.model_weights_file)
#         logging.info("Saved model weights to %s", self.model_weights_file)
#
#
#     def load_model(self):
#         print(self.model_file, self.model_weights_file)
#         # load json and create model
#         json_file = open(self.model_file, 'r')
#         loaded_model_json = json_file.read()
#         json_file.close()
#         loaded_model = model_from_json(loaded_model_json)
#
#         # load weights into new model
#         loaded_model.load_weights(self.model_weights_file)
#
#         logging.info("Loaded model %s, with weights %s", self.model_file, self.model_weights_file)
#
#         return loaded_model
#
#
#     def generate_train_eval_data(self, exclude_rows_from_end = 30, column = "open"):
#         company_list = self.db.get_list_companies()
#         total_companies = self.db.get_companies_count()
#
#         max_items = self.db.get_max_rows() - exclude_rows_from_end
#
#         values = np.zeros((total_companies, max_items))
#
#         i = 0
#         for k in company_list:
#             values_fetch, _ = self.db.get_values_company(company_sym=k, columns=column)
#             values_fetch = values_fetch[:-exclude_rows_from_end]
#             values[i, max_items - len(values_fetch):max_items] = values_fetch
#             i += 1
#
#         total_samples_avail  = max_items - self.days_per_sample
#         # train_sample_size = int(train_partiion*total_samples_avail)
#
#         random_samples = random.sample(range(0, total_samples_avail), total_samples_avail)
#
#         # random_samples = [k for k in range(total_samples_avail)]
#
#         train_samples = random_samples[:total_samples_avail-self.batch_size]
#         eval_samples = random_samples[self.batch_size:]
#
#         if len(eval_samples) < self.days_per_sample:
#             logging.error("Not enough samples to create valuation set, only %s available, require %s: Exiting", len(eval_samples), self.days_per_sample)
#             exit()
#
#         train_iter = 0
#         eval_iter = 0
#
#         epoch_iter = 0
#         batch_count = 0
#
#         th_change = 0.0
#
#         x_train, y_train = None, None
#         x_eval, y_eval = None, None
#
#         import math
#         # custom function
#         def sigmoid(x):
#             return 1 / (1 + math.exp(-x))
#
#         def softmax(x):
#             """Compute softmax values for each sets of scores in x."""
#             e_x = np.exp(x - np.max(x))
#             return e_x / e_x.sum()
#
#         # define vectorized sigmoid
#         sigmoid_v = np.vectorize(sigmoid)
#
#         while True:
#
#             if epoch_iter >= self.epochs:
#                 logging.info("Max epochs reached: Exiting")
#                 raise StopIteration
#
#             if train_iter >= len(train_samples):
#                 random_samples = random.sample(range(0, total_samples_avail), total_samples_avail)
#                 train_samples = random_samples[:total_samples_avail - self.batch_size]
#                 eval_samples = random_samples[self.batch_size:]
#
#                 epoch_iter += 1
#                 train_iter = 0
#                 eval_iter = 0
#
#             eval_iter = train_iter if eval_iter >= len(eval_samples) else eval_iter
#
#             if x_train is None:
#
#                 temp_sample = values[:, train_samples[train_iter]:train_samples[train_iter] + self.days_per_sample + 1]
#
#                 x_train = ((temp_sample[:, 1:-2] - temp_sample[:, 0:-3])/temp_sample[:, 0:-3]).reshape((self.days_per_sample-2) * total_companies)
#                 x_train[np.isnan(x_train)] = 0
#                 x_train[np.isinf(x_train)] = 0
#
#                 y_train = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
#                 y_train[np.isnan(y_train)] = 0
#                 y_train[np.isinf(y_train)] = 0
#
#                 y_train = (y_train - np.min(y_train))/(np.max(y_train) - np.min(y_train))
#
#                 # n_max = np.argmax(y_train)
#                 #
#                 # y_train = y_train * 0
#                 # y_train[n_max] = 1
#                 # y_train[y_train < th_change] = 0
#                 # y_train[y_train > th_change] = 1
#
#
#                 temp_sample = values[:, eval_samples[eval_iter]:eval_samples[eval_iter] + self.days_per_sample + 1]
#
#                 x_eval = ((temp_sample[:, 1:-2] - temp_sample[:, 0:-3])/temp_sample[:, 0:-3]).reshape((self.days_per_sample-2) * total_companies)
#                 y_eval = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
#                 y_eval[np.isnan(y_eval)] = 0
#                 y_eval[np.isinf(y_eval)] = 0
#
#                 # n_max = np.argmax(y_eval)
#                 #
#                 # y_eval = y_eval * 0
#                 # y_eval[n_max] = 1
#
#                 # y_eval[y_eval < th_change] = 0
#                 # y_eval[y_eval > th_change] = 1
#                 # y_eval = softmax(y_eval)
#                 y_eval = (y_eval - np.min(y_eval)) / (np.max(y_eval) - np.min(y_eval))
#
#                 train_iter += 1
#                 eval_iter += 1
#                 batch_count += 1
#
#             else:
#                 if batch_count >= self.batch_size:
#                     x_train = x_train.reshape(self.batch_size, total_companies, self.days_per_sample-2)
#                     y_train = y_train.reshape(self.batch_size, self.companies_count)
#
#                     x_eval = x_eval.reshape(self.batch_size, total_companies, self.days_per_sample-2)
#                     y_eval = y_eval.reshape(self.batch_size, self.companies_count)
#
#                     yield x_train, y_train, x_eval, y_eval
#                     x_train, y_train, x_eval, y_eval = None, None, None, None
#                     batch_count = 0
#                     train_iter += 1
#                     eval_iter += 1
#
#                     continue
#
#
#                 temp_sample = values[:, train_samples[train_iter]:train_samples[train_iter] + self.days_per_sample + 1]
#                 temp_samplex = ((temp_sample[:, 1:-2] - temp_sample[:, 0:-3])/temp_sample[:, 0:-3]).reshape((self.days_per_sample-2) * total_companies)
#                 temp_samplex[np.isnan(temp_samplex)] = 0
#                 temp_samplex[np.isinf(temp_samplex)] = 0
#                 x_train = np.vstack((x_train, temp_samplex))
#
#
#                 temp_sampley = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
#
#                 temp_sampley[np.isnan(temp_sampley)] = 0
#                 temp_sampley[np.isinf(temp_sampley)] = 0
#
#                 # n_max = np.argmax(temp_sampley)
#                 #
#                 # temp_sampley = temp_sampley * 0
#                 # temp_sampley[n_max] = 1
#
#                 # temp_sampley[temp_sampley < th_change] = 0
#                 # temp_sampley[temp_sampley > th_change] = 1
#                 # temp_sampley = softmax(temp_sampley)
#                 temp_sampley = (temp_sampley - np.min(temp_sampley)) / (np.max(temp_sampley) - np.min(temp_sampley))
#
#                 y_train = np.append(y_train, temp_sampley)
#
#                # y_train[y_train < th_change] = 0
#                 # y_train[y_train > th_change] = 1
#
#
#                 # y_train = y_train / np.max(y_train)
#
#                 temp_sample = values[:, eval_samples[eval_iter]:eval_samples[eval_iter] + self.days_per_sample + 1]
#                 temp_samplex = ((temp_sample[:, 1:-2] - temp_sample[:, 0:-3])/temp_sample[:, 0:-3]).reshape((self.days_per_sample-2) * total_companies)
#                 temp_samplex[np.isnan(temp_samplex)] = 0
#                 temp_samplex[np.isinf(temp_samplex)] = 0
#
#
#                 x_eval = np.vstack((x_eval, temp_samplex))
#
#                 temp_sampley = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
#                 temp_sampley[np.isnan(temp_sampley)] = 0
#                 temp_sampley[np.isinf(temp_sampley)] = 0
#
#                 temp_sampley = (temp_sampley - np.min(temp_sampley)) / (np.max(temp_sampley) - np.min(temp_sampley))
#                 y_eval = np.append(y_eval, temp_sampley)
#
#                 # y_eval[y_eval < 0] = 0
#                 # y_eval = y_eval / np.max(y_eval)
#
#                 # y_eval[y_eval < th_change] = 0
#                 # y_eval[y_eval > th_change] = 1
#                 # y_eval = softmax(y_eval)
#
#                 # n_max = np.argmax(y_eval)
#                 #
#                 # y_eval = y_eval * 0
#                 # y_eval[n_max] = 1
#
#                 batch_count += 1
#                 train_iter += 1
#                 eval_iter += 1
#
#
#     def generate_eval_data(self, column = "open"):
#         company_list = self.db.get_list_companies()
#         total_companies = self.db.get_companies_count()
#
#         max_items = self.db.get_max_rows() - 1
#
#         values = np.zeros((total_companies, max_items))
#
#         i = 0
#         for k in company_list:
#             if i == 0:
#                 _, dates_fetch = self.db.get_values_company(company_sym=k)
#             values_fetch, _ = self.db.get_values_company(company_sym=k, columns=column)
#             values_fetch = values_fetch[:-1]
#             values[i, max_items - len(values_fetch):max_items] = values_fetch
#             i += 1
#
#         dates = dates_fetch[-self.days_to_eval:]
#
#         iter = max_items - self.days_per_sample - self.days_to_eval - 1
#
#         x_train, y_train, Dates = None, None, None
#         i = 0
#         while True:
#
#             if x_train is None:
#
#                 temp_sample = values[:, iter:iter + self.days_per_sample + 1]
#
#                 x_train = ((temp_sample[:, 1:-2] - temp_sample[:, 0:-3])/temp_sample[:, 0:-3]).reshape((self.days_per_sample-2) * total_companies)
#                 x_train[np.isnan(x_train)] = 0
#                 x_train[np.isinf(x_train)] = 0
#
#                 y_train = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
#                 y_train[np.isnan(y_train)] = 0
#                 y_train[np.isinf(y_train)] = 0
#
#
#                 iter += 1
#
#             else:
#                 if iter + self.days_per_sample + 1 >= max_items:
#                     x_train = x_train.reshape(self.days_to_eval, total_companies, self.days_per_sample-2)
#                     y_train = y_train.reshape(self.days_to_eval, self.companies_count)
#
#                     return x_train, y_train, dates
#
#                 temp_sample = values[:, iter:iter + self.days_per_sample + 1]
#                 temp_samplex = ((temp_sample[:, 1:-2] - temp_sample[:, 0:-3])/temp_sample[:, 0:-3]).reshape((self.days_per_sample-2) * total_companies)
#                 temp_samplex[np.isnan(temp_samplex)] = 0
#                 temp_samplex[np.isinf(temp_samplex)] = 0
#                 x_train = np.vstack((x_train, temp_samplex))
#
#                 temp_sampley = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
#                 temp_sampley[np.isnan(temp_sampley)] = 0
#                 temp_sampley[np.isinf(temp_sampley)] = 0
#
#                 y_train = np.append(y_train, temp_sampley)
#
#                 iter += 1
#
#
#     def generate_pred_data(self, column = "open"):
#         company_list = self.db.get_list_companies()
#         total_companies = self.db.get_companies_count()
#
#         max_items = self.db.get_max_rows()
#
#         values = np.zeros((total_companies, max_items))
#
#         i = 0
#         for k in company_list:
#             values_fetch , _ = self.db.get_values_company(company_sym=k, columns=column)
#             values[i, max_items - len(values_fetch):max_items] = values_fetch
#             i += 1
#
#         x_train = values[:, -self.days_per_sample:]
#         x_train.reshape(1, total_companies, self.days_per_sample)
#
#         return x_train
#
#     def do_train_and_eval(self):
#         if self.tensorboard_delete:
#             logging.info("FLAG tensorboard_delete is True, Deleting existing files")
#             shutil.rmtree(self.tensorboard_dir)
#
#
#         self.points_model = self.compile_model(self.points_model)
#
#         #setting training data to everything except last n days
#         data_vol = self.generate_train_eval_data(exclude_rows_from_end=self.training_exclude_days, column="volume")
#         data = self.generate_train_eval_data(exclude_rows_from_end=self.training_exclude_days)
#
#         while True:
#             try:
#                 x_t, _, x_e, _ = next(data_vol)
#                 _, y_t, _, y_e = next(data)
#
#                 # x_t = (x_t - np.min(x_t))/(np.max(x_t)-np.min(x_t))
#                 # x_e = (x_e - np.min(x_e))/ (np.max(x_e) - np.min(x_e))
#
#                 # y_t = (y_t - np.min(y_t)) / (np.max(y_t) - np.min(y_t))
#                 # y_e = (y_e - np.min(y_e)) / (np.max(y_e) - np.min(y_e))
#
#                 x_t = np.swapaxes(x_t, 1, 2)
#
#                 i = 0
#                 for k in x_t:
#
#                     j = 0
#                     for coulmn in k:
#                         max_value = np.max(coulmn)
#
#                         if max_value:
#                             x_t[i, j] = x_t[i, j] / max_value
#
#                         j += 1
#                     i += 1
#                 x_t = np.swapaxes(x_t, 2, 1)
#                 # i = 0
#                 # for k in x_t:
#                 #     max_value = np.max(x_t[i, :])
#                 #     x_t[i, :] = x_t[i, :] / max_value
#                 #     i += 1
#
#                 x_t = x_t.reshape(self.batch_size, self.db.get_companies_count(), self.days_per_sample-2, 1)
#                 y_t = y_t.reshape(self.batch_size, -1)
#
#                 x_e = np.swapaxes(x_e, 2, 1)
#                 i = 0
#                 for k in x_e:
#                     j = 0
#                     for coulmn in k:
#                         max_value = np.max(coulmn)
#                         if max_value:
#                             x_e[i, j] = x_e[i, j] / max_value
#                         j += 1
#                     i += 1
#
#                 x_e = np.swapaxes(x_e, 1, 2)
#                 # i = 0
#                 # for k in x_e:
#                 #     max_value = np.max(x_t[i, :])
#                 #     x_e[i, :] = x_e[i, :] / max_value
#                 #     i += 1
#
#                 x_e = x_e.reshape(self.batch_size, self.db.get_companies_count(), self.days_per_sample-2, 1)
#                 y_e = y_e.reshape(self.batch_size, -1)
#
#
#
#                 self.points_model.fit(x_t, y_t, batch_size=self.batch_size, epochs = self.epochs_per_batch, validation_data=(x_e, y_e), callbacks=[self.tensorboard])
#                 # predicted = self.points_model.predict(x_e)
#                 # print(predicted)
#
#                 # exit()
#             except StopIteration:
#                 #Save Model
#                 logging.info("End Training!")
#                 self.save_model(self.points_model)
#                 break
#
#
#     def get_values_for_dates(self, date, column):
#         companies = self.db.get_list_companies()
#
#         values = []
#
#         for sym in companies:
#             fetch = self.db.get_values_company_by_date(sym, date, columns=column)[0]
#             if fetch:
#                 values += fetch
#             else:
#                 values += [0]
#
#
#         return values
#
#
#     def do_eval(self):
#
#         self.points_model = self.load_model()
#         self.points_model = self.compile_model(self.points_model)
#
#         x_, _, d_ = self.generate_eval_data(column="volume")
#         _, y_, _ = self.generate_eval_data()
#         i = 0
#         for k in x_:
#             j = 0
#             for coulmn in k:
#                 max_value = np.max(coulmn)
#                 if max_value:
#                     x_[i, j] = x_[i, j] / max_value
#                 j += 1
#             i += 1
#
#
#         x_ = x_.reshape(self.days_to_eval, self.db.get_companies_count(), self.days_per_sample-2, 1)
#         y_ = y_.reshape(self.days_to_eval, -1)
#
#
#         predictions = self.points_model.predict(x_)
#
#         import matplotlib.pyplot as plt
#
#         for pred, yiter in zip(predictions, y_):
#             print(pred)
#
#             sortedargs = np.argsort(pred)
#
#             yed = yiter[sortedargs]
#
#             # N = 10
#             # conved = np.convolve(yed, np.ones((N,)) / N, mode='valid')
#
#             print(sortedargs[:5], yed[:5], "------------------", np.mean(yed[:5]), np.mean(yiter))
#             plt.plot(yed)
#             plt.plot(np.ones(len(yed))* np.mean(yiter))
#             plt.show()
#             #
#             #
#             # print("----------------------------------------------------------")
#
#         exit()
#         output = dict()
#         output['companies'] = self.db.get_list_companies()
#         output['dates'] = d_
#         output['prediction'] = list(predictions.reshape(self.days_to_eval, self.companies_count))
#         output['ground_truth'] = list(y_.reshape(self.days_to_eval, self.companies_count))
#
#
#
#         output["costs"] = dict()
#         for date in output['dates']:
#             output["costs"][date] = self.get_values_for_dates(date, "open")
#
#
#         self.save_eval_output(output)
#
#     def save_eval_output(self, data):
#
#         try:
#             os.makedirs(self.eval_dir)
#         except OSError:
#             logging.warning("Creation of the directory %s failed" % self.eval_dir)
#         else:
#             logging.info("Successfully created the directory %s " % self.eval_dir)
#
#         logging.info("Writing evaluation output to %s", self.eval_file)
#
#         with open(self.eval_file, 'wb') as outfile:
#             pickle.dump(data, outfile)
#
#         outfile.close()
#
#     def save_pred_output(self, data):
#
#         try:
#             os.makedirs(self.pred_dir)
#         except OSError:
#             logging.warning("Creation of the directory %s failed" % self.pred_dir)
#         else:
#             logging.info("Successfully created the directory %s " % self.pred_dir)
#
#         logging.info("Writing evaluation output to %s", self.pred_dir)
#
#         with open(self.pred_file, 'wb') as outfile:
#             pickle.dump(data, outfile)
#
#         outfile.close()
#
#
#     def do_pred(self):
#         self.points_model = self.load_model()
#         self.compile_model(self.points_model)
#
#         x_ = self.generate_pred_data(column="volume")
#
#         i = 0
#         for k in x_:
#             j = 0
#             for coulmn in k:
#                 max_value = np.max(coulmn)
#                 if max_value:
#                     x_[i, j] = x_[i, j] / max_value
#                 j += 1
#             i += 1
#
#         x_ = x_.reshape(1, self.db.get_companies_count(), self.days_per_sample, 1)
#
#         prediction = self.points_model.predict(x_)
#
#
#         output = dict()
#         output['prediction'] = list(prediction.reshape(self.companies_count))
#
#         self.save_pred_output(output)
#
#         print(output)

# class lstm_ranks(FEModel):
#
#     def __init__(self, args):
#         self.name = self.__class__.__name__
#         self.req_args = []
#         self.opt_args = ['ouput_dir', 'days_per_sample', 'batch_size', 'epochs', 'epoch_per_batch', 'training_exclude_days', "tensorboard_delete", "tensorboard_dir", "model_dir", "days_to_eval", 'pred_dir']
#         FEModel.__init__(self, self.name, self.req_args, self.opt_args)
#
#         self.company_list = None
#         self.black_list = None
#         self.db_folder = None
#         self.output_dir = None
#
#         self.days_per_sample = None
#         self.batch_size = None
#         self.epochs = None
#         self.epochs_per_batch = None
#
#         self.db = None
#
#         self.input_shape = None
#         self.points_model = None
#
#         self.model_dir = None
#
#         self.model_file = None
#         self.model_weights_file = None
#
#
#     def do_init(self, args):
#         self.output_dir = args["output_dir"] if "output_dir" in args.keys() else "Output/"
#
#         self.days_per_sample = int(args["days_per_sample"]) if "days_per_sample" in args.keys() else 200
#         self.batch_size = args["batch_size"] if "batch_size" in args.keys() else 30
#         self.epochs = int(args["epochs"]) if "epochs" in args.keys() else 10
#         self.epochs_per_batch = args["epochs_per_batch"] if "epochs_per_batch" in args.keys() else 1
#
#         self.training_exclude_days = args["training_exclude_days"] if "training_exclude_days" in args.keys() else 30
#         self.tensorboard_dir = args["tensorboard_dir"] if "tensorboard_dir" in args.keys() else self.output_dir+"tensorboard_logs/"+self.name
#         self.tensorboard_delete = args["tensorboard_delete"] if "tensorboard_delete" in args.keys() else True
#         self.model_dir = args["model_dir"] if "model_dir" in args.keys() else self.output_dir+"training_dir/"+self.name+"/"
#         self.eval_dir = args["eval_dir"] if "eval_dir" in args.keys() else self.output_dir+"eval_dir/"+self.name+"/"
#         self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else self.output_dir + "pred_dir/" + self.name + "/"
#
#         self.model_file = os.path.join(os.path.dirname(self.model_dir), "model.json")
#         self.model_weights_file = os.path.join(os.path.dirname(self.model_dir), "model.h5")
#         self.eval_file = os.path.join(os.path.dirname(self.eval_dir), "eval.json")
#         self.pred_file = os.path.join(os.path.dirname(self.pred_dir), "pred.json")
#
#
#         #Evaluation Params
#         self.days_to_eval = args["days_to_eval"] if "days_to_eval" in args.keys() else 30
#
#         logging.info("Test")
#         logging.info("Initializeing, with params: %s", str([k for k in args.items()]))
#
#         self.db = DB_Ops()
#
#         self.companies_count = self.db.get_companies_count()
#         self.input_shape = (self.days_per_sample-2, self.companies_count)
#
#         self.points_model = self.model_init(self.input_shape)
#         self.tensorboard = TensorBoard(log_dir=self.tensorboard_dir, update_freq=10000)
#
#
#     def model_init(self, input_shape):
#         lstm_model = Sequential()
#         lstm_model.add(
#             LSTM(100, input_shape=self.input_shape, batch_size=self.batch_size, dropout=0.0, recurrent_dropout=0.0,
#                  stateful=True, kernel_initializer='random_uniform', return_sequences=True))
#
#         lstm_model.add(LSTM(500, return_sequences=True))
#         lstm_model.add(LSTM(1000))
#         lstm_model.add(Dropout(0.5))
#         lstm_model.add(Dense(2000, activation='relu'))
#         lstm_model.add(Dense(1000, activation='relu'))
#         lstm_model.add(Dense(750, activation='relu'))
#         lstm_model.add(Dense(self.companies_count, activation='softmax'))
#
#         lstm_model.summary()
#
#         return lstm_model
#
#
#     def compile_model(self, model):
#         # optimizer = optimizers.RMSprop(lr=0.01)
#         model.compile(loss='categorical_crossentropy',
#                       optimizer='adam',
#                       metrics=['accuracy'])
#
#         return model
#
#     def save_model(self, model):
#         # Create Folder
#         try:
#             os.makedirs(self.model_dir)
#         except OSError:
#             logging.warning("Creation of the directory %s failed" % self.model_dir)
#         else:
#             logging.info("Successfully created the directory %s " % self.model_dir)
#
#         model_json = model.to_json()
#
#         logging.info("Saved model to %s", self.model_file)
#         with open(self.model_file, "w") as json_file:
#             json_file.write(model_json)
#
#         model.save_weights(self.model_weights_file)
#         logging.info("Saved model weights to %s", self.model_weights_file)
#
#     def load_model(self):
#         print(self.model_file, self.model_weights_file)
#         # load json and create model
#         json_file = open(self.model_file, 'r')
#         loaded_model_json = json_file.read()
#         json_file.close()
#         loaded_model = model_from_json(loaded_model_json)
#
#         # load weights into new model
#         loaded_model.load_weights(self.model_weights_file)
#
#         logging.info("Loaded model %s, with weights %s", self.model_file, self.model_weights_file)
#
#         return loaded_model
#
#
#     def generate_train_eval_data(self, exclude_rows_from_end = 30, column = "volume"):
#         company_list = self.db.get_list_companies()
#         total_companies = self.db.get_companies_count()
#
#         max_items = self.db.get_max_rows() - exclude_rows_from_end
#
#         values = np.zeros((total_companies, max_items))
#
#         i = 0
#         for k in company_list:
#             values_fetch, _ = self.db.get_values_company(company_sym=k, columns=column)
#             values_fetch = values_fetch[:-exclude_rows_from_end]
#             values[i, max_items - len(values_fetch):max_items] = values_fetch
#             i += 1
#
#         total_samples_avail  = max_items - self.days_per_sample
#         # train_sample_size = int(train_partiion*total_samples_avail)
#         #
#         # random_samples = random.sample(range(0, total_samples_avail), total_samples_avail)
#
#         random_samples = [k for k in range(total_samples_avail)]
#
#         train_samples = random_samples[:total_samples_avail-self.batch_size]
#         eval_samples = random_samples[self.batch_size:]
#
#         if len(eval_samples) < self.days_per_sample:
#             logging.error("Not enough samples to create valuation set, only %s available, require %s: Exiting", len(eval_samples), self.days_per_sample)
#             exit()
#
#         train_iter = 0
#         eval_iter = 0
#
#         epoch_iter = 0
#         batch_count = 0
#
#         x_train, y_train = None, None
#         x_eval, y_eval = None, None
#         while True:
#
#             if epoch_iter >= self.epochs:
#                 logging.info("Max epochs reached: Exiting")
#                 raise StopIteration
#
#             if train_iter >= len(train_samples):
#                 epoch_iter += 1
#                 train_iter = 0
#                 eval_iter = 0
#
#             eval_iter = train_iter if eval_iter >= len(eval_samples) else eval_iter
#
#             if x_train is None:
#
#                 temp_sample = values[:, train_samples[train_iter]:train_samples[train_iter] + self.days_per_sample + 1]
#
#                 np.diff(temp_sample, axis=-1)/temp_sample[:,1:]
#
#
#                 # x_train = temp_sample[:, :-1].reshape((self.days_per_sample) * total_companies)
#
#                 x_train = ((temp_sample[:, 1:-2] - temp_sample[:, 0:-3]) / temp_sample[:, 0:-3]).reshape(
#                     (self.days_per_sample - 2) * total_companies)
#                 x_train[np.isnan(x_train)] = 0
#                 x_train[np.isinf(x_train)] = 0
#
#                 y_train = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
#                 y_train[np.isnan(y_train)] = 0
#                 y_train[np.isinf(y_train)] = 0
#
#                 # y_train[y_train < 0] = 0
#                 # y_train[y_train > 0] = 1
#                 # y_train = (y_train - np.min(y_train))/(np.max(y_train) - np.min(y_train))
#                 y_train = np.argsort(y_train)
#
#                 temp_sample = values[:, eval_samples[eval_iter]:eval_samples[eval_iter] + self.days_per_sample + 1]
#
#                 # x_eval = temp_sample[:, :-1].reshape((self.days_per_sample) * total_companies)
#
#                 x_eval = ((temp_sample[:, 1:-2] - temp_sample[:, 0:-3]) / temp_sample[:, 0:-3]).reshape(
#                     (self.days_per_sample - 2) * total_companies)
#                 x_eval[np.isnan(x_eval)] = 0
#                 x_eval[np.isinf(x_eval)] = 0
#
#                 y_eval = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
#                 y_eval[np.isnan(y_eval)] = 0
#                 y_eval[np.isinf(y_eval)] = 0
#
#                 # y_eval[y_eval < 0] = 0
#                 # y_eval[y_eval > 0] = 1
#
#                 # y_eval = (y_eval - np.min(y_eval)) / (np.max(y_eval) - np.min(y_eval))
#                 y_eval = np.argsort(y_eval)
#
#                 train_iter += 1
#                 eval_iter += 1
#                 batch_count += 1
#
#             else:
#                 if batch_count >= self.batch_size:
#                     x_train = x_train.reshape(self.batch_size, self.days_per_sample-2, total_companies)
#                     y_train = y_train.reshape(self.batch_size, self.companies_count)
#
#                     x_eval = x_eval.reshape(self.batch_size, self.days_per_sample-2, total_companies)
#                     y_eval = y_eval.reshape(self.batch_size, self.companies_count)
#                     # print(x_train.shape)
#                     yield x_train, y_train, x_eval, y_eval
#                     x_train, y_train, x_eval, y_eval = None, None, None, None
#                     batch_count = 0
#                     train_iter += 1
#                     eval_iter += 1
#
#                     continue
#
#
#                 temp_sample = values[:, train_samples[train_iter]:train_samples[train_iter] + self.days_per_sample + 1]
#                 # temp_samplex = temp_sample[:, :-1].reshape((self.days_per_sample) * total_companies)
#                 temp_samplex = ((temp_sample[:, 1:-2] - temp_sample[:, 0:-3]) / temp_sample[:, 0:-3]).reshape(
#                     (self.days_per_sample - 2) * total_companies)
#                 temp_samplex[np.isnan(temp_samplex)] = 0
#                 temp_samplex[np.isinf(temp_samplex)] = 0
#
#
#                 x_train = np.vstack((x_train, temp_samplex))
#
#
#                 temp_sampley = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
#
#                 temp_sampley[np.isnan(temp_sampley)] = 0
#                 temp_sampley[np.isinf(temp_sampley)] = 0
#
#                 # temp_sampley = (temp_sampley - np.min(temp_sampley)) / (np.max(temp_sampley) - np.min(temp_sampley))
#
#                 temp_sampley = np.argsort(temp_sampley)
#
#                 # temp_sampley[temp_sampley < 0] = 0
#                 # temp_sampley[temp_sampley > 0] = 1
#
#                 y_train = np.append(y_train, temp_sampley)
#
#                 # y_train[y_train < 0] = 0
#                 # y_train[y_train > 0] = 1
#
#                 temp_sample = values[:, eval_samples[eval_iter]:eval_samples[eval_iter] + self.days_per_sample + 1]
#                 # temp_samplex = temp_sample[:, :-1].reshape((self.days_per_sample) * total_companies)
#                 temp_samplex = ((temp_sample[:, 1:-2] - temp_sample[:, 0:-3]) / temp_sample[:, 0:-3]).reshape(
#                     (self.days_per_sample - 2) * total_companies)
#                 temp_samplex[np.isnan(temp_samplex)] = 0
#                 temp_samplex[np.isinf(temp_samplex)] = 0
#
#                 x_eval = np.vstack((x_eval, temp_samplex))
#
#                 temp_sampley = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
#                 temp_sampley[np.isnan(temp_sampley)] = 0
#                 temp_sampley[np.isinf(temp_sampley)] = 0
#
#                 # temp_sampley = (temp_sampley - np.min(temp_sampley)) / (np.max(temp_sampley) - np.min(temp_sampley))
#
#                 temp_sampley = np.argsort(temp_sampley)
#
#                 # temp_sampley[temp_sampley < 0] = 0
#                 # temp_sampley[temp_sampley > 0] = 1
#
#                 y_eval = np.append(y_eval, temp_sampley)
#
#                 # y_eval[y_eval < 0] = 0
#                 # y_eval[y_eval > 0] = 1
#
#                 batch_count += 1
#                 train_iter += 1
#                 eval_iter += 1
#
#     def generate_eval_data(self, column = "open"):
#         company_list = self.db.get_list_companies()
#         total_companies = self.db.get_companies_count()
#
#         max_items = self.db.get_max_rows() - 1
#
#         values = np.zeros((total_companies, max_items))
#
#         i = 0
#         for k in company_list:
#             if i == 0:
#                 _, dates_fetch = self.db.get_values_company(company_sym=k)
#             values_fetch, _ = self.db.get_values_company(company_sym=k, columns=column)
#             values_fetch = values_fetch[:-1]
#             values[i, max_items - len(values_fetch):max_items] = values_fetch
#             i += 1
#
#         dates = dates_fetch[-self.days_to_eval:]
#
#         iter = max_items - self.days_per_sample - self.days_to_eval - 1
#
#         x_train, y_train, Dates = None, None, None
#         i = 0
#         while True:
#
#             if x_train is None:
#
#                 temp_sample = values[:, iter:iter + self.days_per_sample + 1]
#
#                 x_train = ((temp_sample[:, 1:-2] - temp_sample[:, 0:-3])/temp_sample[:, 0:-3]).reshape((self.days_per_sample-2) * total_companies)
#                 x_train[np.isnan(x_train)] = 0
#                 x_train[np.isinf(x_train)] = 0
#
#                 y_train = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
#                 y_train[np.isnan(y_train)] = 0
#                 y_train[np.isinf(y_train)] = 0
#
#
#                 iter += 1
#
#             else:
#                 if iter + self.days_per_sample + 1 >= max_items:
#                     x_train = x_train.reshape(self.days_to_eval, total_companies, self.days_per_sample-2)
#                     y_train = y_train.reshape(self.days_to_eval, self.companies_count)
#
#                     return x_train, y_train, dates
#
#                 temp_sample = values[:, iter:iter + self.days_per_sample + 1]
#                 temp_samplex = ((temp_sample[:, 1:-2] - temp_sample[:, 0:-3])/temp_sample[:, 0:-3]).reshape((self.days_per_sample-2) * total_companies)
#                 temp_samplex[np.isnan(temp_samplex)] = 0
#                 temp_samplex[np.isinf(temp_samplex)] = 0
#                 x_train = np.vstack((x_train, temp_samplex))
#
#                 temp_sampley = (temp_sample[:, -1] - temp_sample[:, -2]) / temp_sample[:, -2]
#                 temp_sampley[np.isnan(temp_sampley)] = 0
#                 temp_sampley[np.isinf(temp_sampley)] = 0
#
#                 y_train = np.append(y_train, temp_sampley)
#
#                 iter += 1
#
#
#     def do_train_and_eval(self):
#         if self.tensorboard_delete:
#             logging.info("FLAG tensorboard_delete is True, Deleting existing files")
#             if os.path.isdir(self.tensorboard_dir):
#                 shutil.rmtree(self.tensorboard_dir)
#
#
#         self.points_model = self.compile_model(self.points_model)
#
#         #setting training data to everything except last n days
#         data_vol = self.generate_train_eval_data(exclude_rows_from_end=self.training_exclude_days, column="volume")
#         data = self.generate_train_eval_data(exclude_rows_from_end=self.training_exclude_days)
#
#         while True:
#             try:
#                 x_t, _, x_e, _ = next(data_vol)
#                 _, y_t, _, y_e = next(data)
#
#                 # i = 0
#                 # for k in x_t:
#                 #     j = 0
#                 #     for coulmn in k:
#                 #         max_value = np.max(coulmn)
#                 #         if max_value:
#                 #             x_t[i, j] = x_t[i, j] / max_value
#                 #         j += 1
#                 #     i += 1
#
#                 x_t = np.swapaxes(x_t, 1, 2)
#
#                 i = 0
#                 for k in x_t:
#
#                     j = 0
#                     for coulmn in k:
#                         max_value = np.max(coulmn)
#
#                         if max_value:
#                             x_t[i, j] = x_t[i, j] / max_value
#
#                         j += 1
#                     i += 1
#                 x_t = np.swapaxes(x_t, 2, 1)
#
#
#
#                 x_t = x_t.reshape(self.batch_size, self.days_per_sample-2, self.db.get_companies_count())
#                 y_t = y_t.reshape(self.batch_size, -1)
#
#                 # i = 0
#                 # for k in x_e:
#                 #     j = 0
#                 #     for coulmn in k:
#                 #         max_value = np.max(coulmn)
#                 #         if max_value:
#                 #             x_e[i, j] = x_e[i, j] / max_value
#                 #         j += 1
#                 #     i += 1
#
#                 x_e = np.swapaxes(x_e, 2, 1)
#                 i = 0
#                 for k in x_e:
#                     j = 0
#                     for coulmn in k:
#                         max_value = np.max(coulmn)
#                         if max_value:
#                             x_e[i, j] = x_e[i, j] / max_value
#                         j += 1
#                     i += 1
#
#                 x_e = np.swapaxes(x_e, 1, 2)
#
#                 x_e = x_e.reshape(self.batch_size, self.days_per_sample-2, self.db.get_companies_count())
#                 y_e = y_e.reshape(self.batch_size, -1)
#
#                 self.points_model.fit(x_t, y_t, batch_size=self.batch_size, epochs = self.epochs_per_batch, validation_data=(x_e, y_e), callbacks=[self.tensorboard])
#                 predicted = self.points_model.predict(x_e)
#                 print(predicted)
#                 #
#                 # exit()
#             except StopIteration:
#                 #Save Model
#                 logging.info("End Training!")
#                 self.save_model(self.points_model)
#                 break




