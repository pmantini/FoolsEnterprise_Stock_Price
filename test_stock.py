from Stock import Stock


stock_name = 'AAPL'
stock = Stock(stock_name)
stock.update_alpha_vantage()
stock.close()
