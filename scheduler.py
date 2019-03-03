from apscheduler.schedulers.blocking import BlockingScheduler

from App_Setup import app as setupapp
from App_batch_update import app as updateapp

setupapp()
updateapp()

sched = BlockingScheduler()

sched.scheduled_job(updateapp, 'cron', hour=15, minute=00)
sched.start()

