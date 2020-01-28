from FE_Investment_Accounts.portfolio import FEPortfolio

import argparse
import json
import inspect

from FE_Investment_Accounts import portfolio

from setup import log_file

import logging

logging.basicConfig(filename=log_file, format='%(filename)s:%(lineno)s %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(log_file)

formatter = logging.Formatter('%(filename)s:%(lineno)s %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
fh.setFormatter(formatter)


def get_classes_in_module(modulename):
    classes = dir(modulename)

    list_model = []
    for c in classes:
        if not c.startswith('__'):
            list_model.append(c)

    return list_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--account',
        type=str,
        help='Name of the Account',
        default='alpaca')

    parser.add_argument(
        '-a',
        '--arg',
        type=json.loads,
        help='A dictinary of arguments',
        default={})

    args = parser.parse_args()

    #Get list of models
    list_models = get_classes_in_module(portfolio)

    #Check if the model exists
    if args.account not in list_models:
        logging.error("Model %s is invalied" , args.model)
        logging.error("Available Models: %s, EXITING!", str(list_models))
        exit()

    #pick the model from args
    for name, obj in inspect.getmembers(portfolio):
        if inspect.isclass(obj):
            if name == args.account:
                portfolio = obj


    portfolio_obj = portfolio(args)

    #Get list of arguments
    model_arg_available = portfolio_obj.get_args()

    #Check if requireguments are avaialble
    required_args_check_pass = True

    for arg in model_arg_available["required"]:

        if arg not in args.arg.keys():
            logging.error("%s is a required arguement", arg)
            required_args_check_pass = False

    if not required_args_check_pass:
        logging.error("Required arguments check failed, Exiting")
        exit()

    #Warn if optional args do not match
    for argument in args.arg.keys():
        if argument not in model_arg_available["required"] and argument not in model_arg_available["optional"]:
            logging.warning("%s is a not a valid optional arguement", argument)

    portfolio_obj.do_init(args.arg)
    portfolio_obj.do_run()

# if __name__ == "__main__":
#     from argparse import ArgumentParser
#
#     parser = ArgumentParser()
#
#     parser.add_argument("-a", "--account", dest="account",
#                         help="specify the name of the invetment account [alpaca]", metavar="ACCOUNT", default="alpaca")
#     parser.add_argument("-l", "--live", dest="live",
#                         help="specify if the account is live or paper", metavar="LIVE", default=0)
#
#
#     args = parser.parse_args()
#
#
#
#
#
#     portfolio.generate_current_state()