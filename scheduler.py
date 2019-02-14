import schedule
import time

from App_Setup import app as setupapp
from App_batch_update import app as updateapp

setupapp()

schedule.every().day.at("8:35").do(updateapp)

while True:
    schedule.run_pending()
    time.sleep(60) # wait one minute