import datetime
import os

import pickle


NEW_DIR_PATH_RECURSIVE = "output/log/"

def save_qmapping_log(log, dir):
    date_label = datetime.datetime.now().strftime('%Y_%m_%d-%H%M%S')
    
    path = NEW_DIR_PATH_RECURSIVE + date_label + "_" + dir + "/"

    os.makedirs(path, exist_ok=True)



    with open(path + "beta_log_" + dir +  ".pkl", "wb") as tf:
        pickle.dump(log.beta_log, tf)

    with open(path + "q_mapping_value_log_" + dir +  ".pkl","wb") as tf:
        pickle.dump(log.q_mapping_value_log, tf)

    with open(path + "scalled_q_mapping_value_log_" + dir +  ".pkl","wb") as tf:
        pickle.dump(log.scaled_q_mapping_value_log, tf)

    if dir == 'vertical':
        with open(path + "q_value_" + dir + ".pkl","wb") as tf:
            pickle.dump(log.vertical_q_value_log, tf)
    elif dir == 'diagonal':
        with open(path + "q_value_" + dir + ".pkl","wb") as tf:
            pickle.dump(log.diagonal_q_value_log, tf)

    with open(path + "coef.pkl", "wb") as tf:
        pickle.dump(log.coef, tf)

    with open(path + "operate_and_action.pkl", "wb") as tf:
        pickle.dump(log.operation_log, tf)

    with open(path + "Qm_value_diff_log.pkl","wb") as tf:
        pickle.dump(log.qmapping_diff_log, tf)
  
    with open(path + "puressure_log.pkl", "wb") as tf:
        pickle.dump(log.puressure_log, tf)

    with open(path + "obs_log.pkl", "wb") as tf:
        pickle.dump(log.observation_log, tf)
    



if __name__ == "__main__":
    new_dir_path_recursive = "output/log/"
    date_label = datetime.datetime.now().strftime('%Y_%m_%d-%H%M%S')
    print(date_label)
    new_dir_path_recursive += date_label
    os.makedirs(new_dir_path_recursive, exist_ok=True)
