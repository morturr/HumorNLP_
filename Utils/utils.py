import argparse
from datetime import datetime


def print_str(s):
    print(f'\n ~~ {s} ~~ \n')


def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_train_samples', type=int)
    parser.add_argument('n_test_samples', type=int)
    parser.add_argument('task', type=str)  # 'hyperparams', 'train'
    return parser.parse_args()


def print_cur_time(status):
    """ this function print the current time to the stdout """
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(status + " Current Time = " + current_time)
