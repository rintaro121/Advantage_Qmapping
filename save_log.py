import datetime
import os

import numpy as np
import pickle

action_dir = ["LEFT", "UP", "RIGHT", "DOWN"]

NEW_DIR_PATH_RECURSIVE = "./output/log/"

def save_qmapping_log(Q_m, log, path):
    #date_label = datetime.datetime.now().strftime('%Y_%m_%d-%H%M%S')
    
    #path = NEW_DIR_PATH_RECURSIVE + date_label + "_" + dir + "/"

    os.makedirs(path, exist_ok=True)

    # puressure log
    with open(path + "puressure_log.pkl", "wb") as tf:
        pickle.dump(log.puressure_log, tf)

    # obs log
    with open(path + "obs_log.pkl", "wb") as tf:
        pickle.dump(log.observation_log, tf)

    # action log
    with open(path + "action_log.pkl", "wb") as tf:
        pickle.dump(log.move_cnt, tf)

    # beta log
    with open(path + "beta_log.pkl","wb") as tf:
        pickle.dump(log.beta_log, tf)




if __name__ == "__main__":
    new_dir_path_recursive = "output/log/"
    date_label = datetime.datetime.now().strftime('%Y_%m_%d-%H%M%S')
    print(date_label)
    new_dir_path_recursive += date_label
    os.makedirs(new_dir_path_recursive, exist_ok=True)
