from Stock import Stock
from stock_api.batch_query_alpha_vantage import Stock_Query
from Stock_List import Stock_List

# stock_list = Stock_List()
# list_of_stocks = stock_list.list_of_stocks()
# batch = 100
# stock_name = [k[0] for k in list_of_stocks]
# test = Stock_Query()
# print(test.query(stock_name))

# k = Stock('ATVI')
# for l in k.fetch_latest(10):
#     print(l)

# k = Stock('AAPL')
# # k.update_alpha_vantage()
# k.update_alpha_vantage()


from stock_api.stock_query_alpha_vantage import Stock_Query

test = Stock_Query()
test.query('AET')