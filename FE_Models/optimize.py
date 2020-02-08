import numpy as np
from scipy.optimize import linprog

class Optimize:

    def linear_problem(self, delta_p, prices, resource=500, number_of_stocks=6):



        c = delta_p

        # c = -c
        A_eq = np.array([np.ones(len(prices))])

        A_ub = [prices]
        b_ub = [resource]

        failed = True
        while failed:
            try:
                b_eq = np.array([number_of_stocks])

                # results = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub)
                results = linprog(c, A_ub=A_ub, b_ub=b_ub)
                failed = False
            except:
                print("ecept")
                number_of_stocks += 1


        res = list(results.x)



        choices = [n for n, i in enumerate(res) if i > 1.]
        qunatities = results.x[results.x > 1]

        return choices, qunatities

    def random_selection(self, predictions, prices, confidence=None, resource=5000, number_of_stocks=6, min_price = 10):
        """ Multiply class values with probabilities, and then minimize risk by spreading out investments"""

        def get_resources(investments=[1500, 1000, 500, 250, 175, 75]):
            investments = [2*(k+1)*resource/((number_of_stocks+1)*number_of_stocks) for k in range(number_of_stocks)]
            print(investments)
            # investments = [4000, 2000]
            for k in reversed(investments):
                yield k

        # delta_p = np.zeros(predictions.shape)
        #
        # delta_p[predictions <= 2] = 0.0
        # delta_p[predictions == 3] = 50
        # delta_p[predictions == 4] = 1000
        # delta_p[predictions == 5] = 10000
        #
        # # stock_prefs = delta_p * confidence
        # stock_prefs = delta_p
        # stock_prefs = (stock_prefs - np.min(stock_prefs)) / (np.max(stock_prefs) - np.min(stock_prefs))

        best_ops = np.argsort(predictions)
        # best_ops = [k for k in range(len(predictions))]
        # print(best_ops)

        choices, quantities = [], []

        resource_gen = get_resources()
        current_resource = next(resource_gen)
        for best in best_ops:
            random_value = np.random.random()
            if prices[best] < min_price:
                continue

            if prices[best] < current_resource:
                choices += [best]
                quantities += [current_resource / prices[best]]
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


