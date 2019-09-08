from scipy.optimize import linprog
import numpy as np

class Strategy:

    def linear_problem(self, predictions, prices, confidence=None, resource=5000):
        # print(predictions)
        delta_p = np.zeros(predictions.shape)
        delta_p[predictions <= 2] = 0
        delta_p[predictions == 3] = 0.025
        delta_p[predictions == 4] = 0.05
        delta_p[predictions == 5] = 0.075

        # delta_p[predictions == 3] = 1
        # delta_p[predictions == 4] = 1
        # delta_p[predictions == 5] = 1

        # bounds = [(None,None) if k <= 2 else (1,None) for k in predictions]

        number_of_stocks = 10
        # c = prices * delta_p * 100
        # c = prices * delta_p
        c = prices * delta_p

        # c = [np.ones(len(prices))] * delta_p
        c = (c - np.min(c))/(np.max(c) - np.min(c))
        c = -c
        A_eq = np.array([np.ones(len(prices))])
        # b_eq = np.array([number_of_stocks])

        A_ub = [prices]
        b_ub = [resource]
        # print(len(c), len(A_ub), len(b_ub))
        failed = True
        while failed:
            try:
                b_eq = np.array([number_of_stocks])
                # results = linprog(c, A_ub=A_ub, b_ub=b_ub)
                results = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub)
                failed = False
            except:
                # print("Re-optmizing")
                number_of_stocks += 1

        # print(results)
        res = list(results.x)

        # print(res)
        # print([ n for n,i in enumerate(res) if i>0. ][0])
        # print([ n for n,i in enumerate(res) if i>0. ], results.x[results.x > 0], prices[results.x > 0])
        choices = [n for n, i in enumerate(res) if i > 1.]
        qunatities = results.x[results.x > 1]

        return choices, qunatities

    def linear_problem2(self, predictions, prices, confidence=None, resource = 5000):
        # print(predictions)
        delta_p = np.zeros(predictions.shape)
        # delta_p[predictions == 0] = -0.05
        # delta_p[predictions == 1] = -0.025
        delta_p[predictions <= 2] = 0.0
        delta_p[predictions == 3] = 0.025
        delta_p[predictions == 4] = 0.025
        delta_p[predictions == 5] = 0.025

        # delta_p[predictions == 3] = 10
        # delta_p[predictions == 4] = 500
        # delta_p[predictions == 5] = 1000


        ##########Original########################
        number_of_stocks = 10
        # c = prices * delta_p * 100
        ##########################best##############################
        c = -delta_p * confidence
        ##########################best##############################

        A_eq = np.array([np.ones(len(prices))])
        b_eq = np.array([number_of_stocks])

        # A_ub = [prices]
        # b_ub = [resource]
        # print(len(c), len(A_ub), len(b_ub))
        ##########Original########################

        # prices1 = np.zeros(prices.shape)
        # prices2 = np.zeros(prices.shape)
        # prices3 = np.zeros(prices.shape)
        # prices4 = np.zeros(prices.shape)
        #
        #
        # prices1[:150] = prices[:150]
        # prices2[150:300] = prices[150:300]
        # prices3[300:450] = prices[300:450]
        # prices4[450:] = prices[450:]
        # # #
        # A_eq = [prices1, prices2, prices3, prices4]
        # b_eq = [2000,2000,2000,2000]
        #
        # A_ub = [prices1, prices2, prices3, prices4]
        # total_value = np.sum(prices)
        # b_ub = [np.sum(k) / total_value * resource for k in [prices1, prices2, prices3, prices4]]

        failed = True

        while failed:
            try:
                # results = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub)
                # b_eq = np.array([number_of_stocks])
                # results = linprog(c, A_ub=A_ub, b_ub=b_ub)
                # results = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub = A_ub, b_ub=b_ub)
                results = linprog(c, A_eq=A_eq, b_eq=b_eq)
                failed = False
            except:
                # print("Re-optmizing")
                # number_of_stocks+= 10
                pass

        # print(results)
        res = list(results.x)

        # print(res)
        # print([ n for n,i in enumerate(res) if i>0. ][0])
        # print([ n for n,i in enumerate(res) if i>0. ], results.x[results.x > 0], prices[results.x > 0])
        choices = [ n for n,i in enumerate(res) if i>1. ]
        qunatities = results.x[results.x > 1]

        return choices, qunatities

    def random_selection(self, predictions, prices, confidence=None, resource = 5000):
        """ Multiply class values with probabilities, and then minimize risk by spreading out investments"""


        def get_resources(investments = [2500, 2000, 1000, 500]):
            investments = [4000, 2000]
            for k in investments:
                yield k



        choices, quantities = [], []

        resource_gen =  get_resources()
        current_resource =  next(resource_gen)
        while True:
            best = int(np.random.random() * len(prices))

            if prices[best] < current_resource:
                choices += [best]
                quantities += [current_resource/prices[best]]
                try:
                    current_resource = next(resource_gen)
                except:
                    break


        return choices, quantities



    def random_selection1(self, predictions, prices, confidence=None, resource = 5000):
        """ Multiply class values with probabilities, and then minimize risk by spreading out investments"""


        def get_resources(investments = [2500, 2000, 1000, 500]):
            # investments = [1000 for k in range(5)]
            investments = [4000, 2000]
            for k in investments:
                yield k

        delta_p = np.zeros(predictions.shape)

        delta_p[predictions <= 2] = 0.0
        delta_p[predictions == 3] = 50
        delta_p[predictions == 4] = 1000
        delta_p[predictions == 5] = 10000

        # stock_prefs = delta_p * confidence
        stock_prefs = delta_p
        stock_prefs = (stock_prefs - np.min(stock_prefs))/(np.max(stock_prefs) - np.min(stock_prefs))

        best_ops = np.argsort(-stock_prefs)
        # print(best_ops)

        choices, quantities = [], []

        resource_gen =  get_resources()
        current_resource =  next(resource_gen)
        for best in best_ops:
            random_value = np.random.random()

            if random_value < confidence[best]:
                if prices[best] < current_resource:
                    choices += [best]
                    quantities += [current_resource/prices[best]]
                    try:
                        current_resource = next(resource_gen)
                    except:
                        break

        # print(choices, quantities)
        # exit()
        #
        #
        # print(best_ops)
        #
        # print(stock_prefs)
        # exit()
        return choices, quantities

