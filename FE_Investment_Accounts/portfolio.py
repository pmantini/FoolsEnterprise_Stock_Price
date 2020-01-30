from FE_Investment_Accounts.alpaca import FE_Alpaca
from FE_Stock.FE_DB_Models.FE_Stock_List import FE_Stock_List
from setup import table_name, mount_folder
import numpy as np
import pickle, os
import sys
from datetime import datetime, timedelta

import logging
import os
import shutil

logging = logging.getLogger("main")


class FEPortfolio:
    def __init__(self, name, req_args, opt_args):
        self.name = name
        self.req_args = req_args
        self.opt_args = opt_args

    def get_args(self):
        return {"required": self.req_args, "optional": self.opt_args}

    def do_run(self):
        pass



class Alpaca(FEPortfolio):

    def __init__(self, args):
        self.name = self.__class__.__name__
        self.req_args = ['strategy', 'live']
        self.opt_args = ['output_dir']
        FEPortfolio.__init__(self, self.name, self.req_args, self.opt_args)

        self.current_state = None
        self.desired_state = None
        self.actions = None

        self.live = int(args.arg["live"])
        self.strategy = args.arg["strategy"]

        self.strategy_actions = None

        self.investment_ac = self.get_investment_account()

        self.assets = self.get_asset_list()

        self.stock_position = {"n": 0,
                               "b": 1,
                               "h": 2,
                               "s": 3,
                               "l": 4
                               }

        self.stock_position_name = {0: "not_holding",
                                   1: "limit_buy",
                                   2: "holding",
                                   3: "limit_sell",
                                   4: "liqidate"
                                   }

        self.model_input_dir = None
        self.list_of_orders = {}


    def do_init(self, args):

        self.input_dir = mount_folder
        self.portfolio_dir = os.path.join(mount_folder, "portfolio")
        self.this_portfolio = os.path.join(self.portfolio_dir, self.name)
        self.account_file = os.path.join(self.this_portfolio, "account.json")

        self.strategy_input_dir = args["strategy_input_dir"] if "strategy_input_dir" in args.keys() else "Output/Strategy"
        self.pred_dir = args["pred_dir"] if "pred_dir" in args.keys() else "pred_dir"
        self.pred_file = args["pred_file"] if "pred_file" in args.keys() else "pred.json"

        self.strategy_file = os.path.join(os.path.join(self.strategy_input_dir, self.pred_dir), os.path.join(self.strategy,self.pred_file))


    def do_save_status(self):
        data = self.investment_ac.get_account_details()

        try:
            os.makedirs(self.this_portfolio)
        except OSError:
            logging.warning("Creation of the directory %s failed" % self.this_portfolio)
        else:
            logging.info("Successfully created the directory %s " % self.this_portfolio)

        logging.info("Writing evaluation output to %s", self.account_file)

        with open(self.account_file, 'wb') as outfile:
            pickle.dump(data, outfile)

        outfile.close()



    def get_asset_list(self):
        stock_list = FE_Stock_List()
        stock_list.init(table_name=table_name)
        list_stocks = stock_list.list_of_stocks()

        return [k[0] for k in list_stocks]



    def get_investment_account(self):

        return FE_Alpaca(self.live)



    def generate_current_state(self):

        self.current_state = np.zeros(len(self.assets))

        orders = self.investment_ac.get_all_order()
        positions = self.investment_ac.get_all_positions()
        list_of_positions = {}
        for k in orders:

            self.list_of_orders[k.symbol] = k

        for k in positions:
            list_of_positions[k.symbol] = k


        for k in enumerate(self.assets):

            if k[1] in self.list_of_orders.keys():
                if self.list_of_orders[k[1]].side == "buy" and self.list_of_orders[k[1]].type == "limit":
                    self.current_state[k[0]] = self.stock_position["b"]

                elif self.list_of_orders[k[1]].side == "sell" and self.list_of_orders[k[1]].type == "limit":
                    self.current_state[k[0]] = self.stock_position["s"]

            elif k[1] in list_of_positions.keys():
                self.current_state[k[0]] = self.stock_position["h"]

        return self.current_state



    def generate_desired_state(self):

        self.desired_state = np.zeros(len(self.assets))

        #load strategy info
        self.strategy_actions = self.load_model(self.strategy_file)

        self.orders = self.investment_ac.get_all_order()
        positions = self.investment_ac.get_all_positions()
        list_of_orders, list_of_positions = {}, {}
        for k in self.orders:
            list_of_orders[k.symbol] = k

        for k in positions:
            list_of_positions[k.symbol] = k


        for k in enumerate(self.assets):

            if k[1] in self.strategy_actions.keys():
                self.desired_state[k[0]] = self.stock_position["b"]


            if k[1] in list_of_positions.keys():
                self.desired_state[k[0]] = self.stock_position["s"]

            if k[1] in list_of_orders.keys():
                if list_of_orders[k[1]].side == "buy":
                    exp_date_str =  list_of_orders[k[1]].client_order_id.split("___")[1]
                    exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d')
                    print(self.assets[k[0]], exp_date)

                    if datetime.today() > exp_date:
                        print("for cancelling")
                        self.desired_state[k[0]] = self.stock_position["n"]
                elif list_of_orders[k[1]].side == "sell":
                    exp_date_str = list_of_orders[k[1]].client_order_id.split("___")[1]
                    exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d')
                    print(self.assets[k[0]], exp_date)
                    if datetime.today() >= exp_date-timedelta(days=1):
                        print("Predicted sell price expiring, Liquidating")
                        self.desired_state[k[0]] = self.stock_position["l"]
                    # if datetime.today() >= exp_date-timedelta(days=2):
                    #     print("for cancelling")
                    #     self.desired_state[k[0]] = self.stock_position["e"]


    def load_model(self, file):
        if str(file).endswith(".json"):

            with open(file, 'rb') as f:
                return pickle.load(f)

        elif str(file).endswith(".npy"):
            return np.load(file)
        else:
            raise Exception("%s does not have a valid file format (json or npy)")

    def do_run(self):
        self.generate_current_state()
        self.generate_desired_state()

        for k  in enumerate(self.current_state):

            if self.current_state[k[0]] != self.desired_state[k[0]]:
                print(self.assets[k[0]], self.stock_position_name[k[1]], self.stock_position_name[self.desired_state[k[0]]])

                this_asset = self.assets[k[0]]
                if self.stock_position_name[self.desired_state[k[0]]] == "limit_buy":


                    try:
                        buy_order_name = this_asset+"___"+self.strategy_actions[this_asset]["buy_exp"]
                        # buy_order_name = this_asset + "___" + (datetime.today() - timedelta(days=2)).strftime(
                        #     '%Y-%m-%d')
                        order_resp = self.investment_ac.order(this_asset, int(self.strategy_actions[this_asset]["quantity"]),
                                                              "buy", self.strategy_actions[this_asset]["buy"][0] - 1, buy_order_name)

                        print(order_resp)
                    except:
                        print("Unexpected error:", sys.exc_info())
                        print("Failed to buy: ", this_asset)
                        pass

                if self.stock_position_name[self.desired_state[k[0]]] == "limit_sell":

                    try:

                        avg_buy_price = self.investment_ac.get_position(this_asset).avg_entry_price
                        new_sell_price = float(avg_buy_price) * float(self.strategy_actions[this_asset]["sell_ratio"])
                        sell_order_name = this_asset + "___" + self.strategy_actions[this_asset]["sell_exp"]
                        # sell_order_name = this_asset + "___" + (datetime.today() + timedelta(days=1)).strftime(
                        #     '%Y-%m-%d')
                        available_qty = self.investment_ac.get_position(this_asset).qty
                        order_resp = self.investment_ac.order(this_asset, int(available_qty), "sell",
                                                 new_sell_price, sell_order_name)
                        print(order_resp)
                    except:
                        print("Unexpected error:", sys.exc_info())
                        print("Failed to buy: ", this_asset)
                        pass

                if self.stock_position_name[self.desired_state[k[0]]] == "not_holding" and \
                        self.stock_position_name[self.current_state[k[0]]] == "limit_buy":
                        order_id = self.list_of_orders[this_asset].id
                        print("Canceling order %s, %s" % (order_id, this_asset))
                        self.investment_ac.cancel_order(order_id)

                if self.stock_position_name[self.desired_state[k[0]]] == "limit_sell" and \
                        self.stock_position_name[self.current_state[k[0]]] == "liquidate":
                        order_id = self.list_of_orders[this_asset].id
                        print("Liqudating asset %s, %s" % (order_id, this_asset))
                        self.investment_ac.cancel_order(order_id)
                        self.investment_ac.liquidate_position(this_asset)


