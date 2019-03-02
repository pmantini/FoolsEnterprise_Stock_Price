from apscheduler.schedulers.blocking import BlockingScheduler
import time

from App_Setup import app as setupapp
from App_batch_update import app as updateapp

# setupapp()
# updateapp()

sched = BlockingScheduler()

sched.scheduled_job(updateapp, 'cron', hour=8, minute=40)

# schedule.every().day.at("8:35:00").do(updateapp)
#
# while True:
#     schedule.run_pending()
#     time.sleep(60) # wait one minute