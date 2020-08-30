from alpaca_trade_api.rest import REST, Order


class MyRest(REST):

    def submit_order_oco(self, symbol, qty, side, type, time_in_force,
                     limit_price_p, stop_price, limit_price_l, client_order_id=None,
                     extended_hours=None):
        '''Request a new order'''
        params = {
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'type': type,
            'time_in_force': time_in_force,
            "order_class": "oco",
            "take_profit": {
                "limit_price": limit_price_p
            },
            "stop_loss": {
                "stop_price": stop_price,
                "limit_price": limit_price_l
            }
        }

        if client_order_id is not None:
            params['client_order_id'] = client_order_id
        if extended_hours is not None:
            params['extended_hours'] = extended_hours
        resp = self.post('/orders', params)
        return Order(resp)
