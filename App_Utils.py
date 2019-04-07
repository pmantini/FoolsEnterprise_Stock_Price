import datetime

def is_update_required(last_date):
    if datetime.datetime.today().strftime('%Y-%m-%d') == last_date:
        return False
    if datetime.datetime.today().weekday() == 5:
        if last_date == (datetime.datetime.today() - datetime.timedelta(1)).strftime('%Y-%m-%d'):
            return False
    if datetime.datetime.today().weekday() == 6:

        if last_date == (datetime.datetime.today() - datetime.timedelta(2)).strftime('%Y-%m-%d'):
            return False

    return True
