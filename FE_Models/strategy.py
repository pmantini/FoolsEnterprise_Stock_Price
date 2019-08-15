from scipy.optimize import linprog
import numpy as np

class Strategy:

    def linear_problem(self, predictions, prices, confidence=None, resource = 5000):
        # print(predictions)
        delta_p = np.zeros(predictions.shape)
        delta_p[predictions <= 2] = 0
        delta_p[predictions == 3] = 0.025
        delta_p[predictions == 4] = 0.05
        delta_p[predictions == 5] = 0.075

        # delta_p[predictions == 3] = 10
        # delta_p[predictions == 4] = 50
        # delta_p[predictions == 5] = 1000


        # bounds = [(None,None) if k <= 2 else (1,None) for k in predictions]

        number_of_stocks = 10
        # c = prices * delta_p * 100
        c = prices * delta_p
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
                results = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub = A_ub, b_ub=b_ub)
                failed = False
            except:
                # print("Re-optmizing")
                number_of_stocks+= 1

        # print(results)
        res = list(results.x)

        # print(res)
        # print([ n for n,i in enumerate(res) if i>0. ][0])
        # print([ n for n,i in enumerate(res) if i>0. ], results.x[results.x > 0], prices[results.x > 0])
        choices = [ n for n,i in enumerate(res) if i>1. ]
        qunatities = results.x[results.x > 1]

        return choices, qunatities

