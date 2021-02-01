import numpy as np
from scipy.optimize import linprog
# from sklearn.linear_model import LinearRegression

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

    def random_selection(self, close_changes, prices, resource=5000, number_of_stocks=6, min_price = 5, dropout = 0.25):

        def get_resources(investments=[1500, 1000, 500, 250, 175, 75]):
            investments = [2*(k+1)*resource/((number_of_stocks+1)*number_of_stocks) for k in range(number_of_stocks)]
            for k in reversed(investments):
                yield k


        best_ops = np.argsort(close_changes)

        choices, quantities = [], []

        resource_gen = get_resources()
        current_resource = next(resource_gen)
        for best in best_ops:
            if close_changes[best] > 0:
                continue
            if prices[best] < min_price:
                continue


            if np.random.random() < dropout:
                print(best, ":Droped out")
                continue

            if prices[best] < current_resource:
                choices += [best]
                quantities += [current_resource / prices[best]]
                try:
                    current_resource = next(resource_gen)
                except:
                    break
        return choices, quantities


    def random_selection_penny(self, close_changes, prices, resource=5000, number_of_stocks=6, dropout = 0.25):

        def get_resources(investments=[1500, 1000, 500, 250, 175, 75]):
            investments = [2*(k+1)*resource/((number_of_stocks+1)*number_of_stocks) for k in range(number_of_stocks)]
            for k in reversed(investments):
                yield k


        best_ops = np.argsort(close_changes)

        choices, quantities = [], []

        resource_gen = get_resources()
        current_resource = next(resource_gen)
        for best in best_ops:
            if close_changes[best] > 0:
                continue
            if prices[best] > 1:
                continue


            if np.random.random() < dropout:
                print(best, ":Droped out")
                continue

            if prices[best] < current_resource:
                choices += [best]
                quantities += [current_resource / prices[best]]
                try:
                    current_resource = next(resource_gen)
                except:
                    break
        return choices, quantities



    # def random_selection(self, predictions, prices, high_prices, confidence=None, resource=5000, number_of_stocks=6, min_price = 5, dropout = 0.25):
    #     """ Multiply class values with probabilities, and then minimize risk by spreading out investments"""
    #
    #     def get_co_efficients(prices, count_from_last):
    #         list_of_prices_low = np.array(prices[-count_from_last:])
    #
    #         values = list_of_prices_low
    #
    #         x, y = np.array([ind for ind in np.arange(5)]).reshape(-1, 1), values.reshape(-1, 1)
    #
    #         model = LinearRegression().fit(x, y)
    #         x_new = np.arange(5).reshape((-1, 1))
    #
    #         # reg_line = model.predict(x_new)
    #
    #         return model.coef_[0][0]
    #
    #     def get_resources(investments=[1500, 1000, 500, 250, 175, 75]):
    #         investments = [2*(k+1)*resource/((number_of_stocks+1)*number_of_stocks) for k in range(number_of_stocks)]
    #         print(investments)
    #         # investments = [4000, 2000]
    #         for k in reversed(investments):
    #             yield k
    #
    #     # delta_p = np.zeros(predictions.shape)
    #     #
    #     # delta_p[predictions <= 2] = 0.0
    #     # delta_p[predictions == 3] = 50
    #     # delta_p[predictions == 4] = 1000
    #     # delta_p[predictions == 5] = 10000
    #     #
    #     # # stock_prefs = delta_p * confidence
    #     # stock_prefs = delta_p
    #     # stock_prefs = (stock_prefs - np.min(stock_prefs)) / (np.max(stock_prefs) - np.min(stock_prefs))
    #
    #     best_ops = np.argsort(predictions)
    #     # best_ops = np.nonzero(predictions)[0]
    #     # np.random.shuffle(best_ops)
    #
    #     # best_ops = [k for k in range(len(predictions))]
    #     # print(best_ops)
    #
    #     choices, quantities = [], []
    #
    #     resource_gen = get_resources()
    #     current_resource = next(resource_gen)
    #     for best in best_ops:
    #
    #         # this_coeff = get_co_efficients(high_prices[best], 5)
    #         # # print(best, this_coeff, predictions[best])
    #         # if abs(this_coeff) > 2:
    #         #     print("Skipping as the coefficent (%s) is > %s (Risky)" % (this_coeff, 2))
    #         #     continue
    #
    #         # random_value = np.random.random()
    #         if prices[best] < min_price:
    #             continue
    #
    #         if np.random.random() < dropout:
    #             print(best, ":Droped out")
    #             continue
    #
    #         if prices[best] < current_resource:
    #             choices += [best]
    #             quantities += [current_resource / prices[best]]
    #             try:
    #                 current_resource = next(resource_gen)
    #             except:
    #                 break
    #
    #     # print(choices, quantities)
    #     # exit()
    #     #
    #     #
    #     # print(best_ops)
    #     #
    #     # print(stock_prefs)
    #     # exit()
    #     return choices, quantities
    #
    #
