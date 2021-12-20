import argparse
import datetime
import os
import random
import sys
import threading
import warnings

import numpy as np
from numpy.core.fromnumeric import argmax
import pickle
import roboschool
import torch
import torch.nn.functional as F
import time
import gym.wrappers
from PIL import Image

from PPO import PPO
from save_log import save_qmapping_log
import wrappers
from q_mapping import Qmapping
from utils.utils import read_sensor_ch, read_CH_and_pressure
import dispnum

warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('dir', default="vertical", help="direction which boards move", choices=['vertical', "diagonal"])
args = parser.parse_args()

DEFAULT_ENV_NAME = "RoboschoolPong-v1"  # Use a longer version of Pong for demonstration (needs to be defined in source)
MAKE_VIDEO = False  # Set true or false here to record video OR render, not both
ACTION_DIRECTION = ["LEFT", "UP", "RIGHT", "DOWN"]
ADDRESS = '/dev/tty.usbmodem14401'

env = gym.make(DEFAULT_ENV_NAME)
env = wrappers.action_space_discretizer(env, 2)
state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std \
        =    13, 4, 0.0003, 0.001, 0.99, 80, 0.2, False, 0.6

net = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
net.load("weight/PPO_RoboschoolPong-v1_0_0.pth")
env.reset()
recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env, "./recording.mp4", enabled=MAKE_VIDEO)
still_open = True

mycam = env.unwrapped.scene.cpp_world.new_camera_free_float(320,200,"my_camera")
mycam.move_and_look_at( 0, 0, 3, 0, 0, 0)

cnt = 1
max_en = 0
min_en = 100

Q_m = Qmapping("weight/PPO_RoboschoolPong-v1_0_0.pth", args)

dir = ["LEFT", "UP", "RIGHT", "DOWN"]

situation_for_calibration = {
    "LEFT" : [],
    "UP" : [],
    "RIGHT" : [],
    "DOWN" : [],
}

disp_num_list = [dispnum.one, dispnum.two, dispnum.three, dispnum.four, dispnum.five]

def show_countdown():
    for i in range(3):
        print(disp_num_list[2-i])
        
        time.sleep(1)
        os.system('clear')    

def calibration(env, obs):
    env = set_state_for_calibration(env, obs)

    for i in range(50):
        operation_ch, puressure_info = read_CH_and_pressure(ADDRESS)
        done = False
        if not operation_ch:
            action = -1
        else:
            action = Q_m.interpret(operation_ch, obs, calibration=True)
            action = Q_m.reversable_interpret(operation_ch, obs, True, i)
        obs, r, done, _ = env.step(np.array(action))

        env.render("human")

        rgb, _, _, _, _ = mycam.render(False, False, False)
        mycam.test_window()


def calibration_and_check_advantage(env, obs):
    env = set_state_for_calibration(env, obs)
    best_action = Q_m.PPO.best_action(obs)


    for i in range(50):
        operation_ch, puressure_info = read_CH_and_pressure(ADDRESS)
        done = False
        
        if not operation_ch:
            action = -1
        else:
            action = Q_m.interpret(operation_ch, obs, calibration=True)
        
        obs, r, done, _ = env.step(np.array(action))

        tmp_best_action = Q_m.PPO.best_action(obs)
        if best_action != tmp_best_action:
            print("end calibration")
            break
        env.render("human")

        rgb, _, _, _, _ = mycam.render(False, False, False)
        mycam.test_window()

def calibration_and_check_input(env, obs):
    env = set_state_for_calibration(env, obs)
    first_operation = None

    for i in range(25):
        env.render("human")

        rgb, _, _, _, _ = mycam.render(False, False, False)
        mycam.test_window() 
        operation_ch, puressure_info = read_CH_and_pressure(ADDRESS)
        #print(first_operation, operation_ch)
        if operation_ch is not None and first_operation is not None and first_operation != operation_ch:
            break
        done = False
        if operation_ch is not None and first_operation is None:
            first_operation = operation_ch
        if not operation_ch:
            action = -1
        else:
            action = Q_m.reversable_interpret(operation_ch, obs, True, i, False)

        obs, r, done, _ = env.step(np.array(action))


def calibration_by_image(env, obs):
    env = set_state_for_calibration(env, obs)
    env.step(np.array(-1))
    env.render("human")
    rgb, _, _, _, _ = mycam.render(False, False, False)
    mycam.test_window() 
    while True:
        operation_ch, puressure_info = read_CH_and_pressure(ADDRESS)
        env.render("human")
        rgb, _, _, _, _ = mycam.render(False, False, False)
        mycam.test_window() 
        if operation_ch is not None:
            action = Q_m.reversable_interpret(operation_ch, obs, True, i)
            break
    time.sleep(2)


def set_state_for_calibration(env, obs):

    env.reset()

    p0x, p0y, bx, bvx, by, bvy = obs[0],obs[2],obs[8],obs[9],obs[10],obs[11]
    env.scene.p0x.reset_current_position(p0x, 0)
    env.scene.p0y.reset_current_position(p0y, 0)

    env.scene.ballx.reset_current_position(bx, bvx)
    env.scene.bally.reset_current_position(by, bvy)

    return env


def detect_Qmapping_error_by_operation(Qmapping : Qmapping, op_ch : int, obs : list) -> bool:
    # 一番いいアクションがq-mappingで割り当てられていない
    # current_qmapping = Qmapping.q_mapping[op_ch]
    advantage_value = Qmapping.calc_advantage(obs)

    most_valuable_dirction_id = np.argmax(advantage_value).item()
    
    res = True
    for _, Qmapping_value in Qmapping.q_mapping.items():
        id = np.argmax(Qmapping_value)
        if most_valuable_dirction_id == id:
            res = False

    return res

def play_pong(cnt, loop_cnt):
    is_finish = True
    obs = env.reset()

    restart_delay = 0
    score = 0
    while True:
        operation_ch, puressure_info = read_CH_and_pressure(ADDRESS)
        done = False
        cnt += 1
        if not operation_ch:
            action = -1
        else:
            action = Q_m.reversable_interpret(operation_ch, obs, calibration=False, frame=cnt, play_only=False)

            
        obs, r, done, _ = env.step(np.array(action))

        env.render("human")

        score += r

        rgb, _, _, _, _ = mycam.render(False, False, False)
        mycam.test_window()
            
        if detect_Qmapping_error_by_operation(Q_m, operation_ch,obs) and cnt >= 300 and loop_cnt < 4:
            is_finish = False
            return is_finish

        if not done: continue

        if restart_delay == 0:
            print("score=%0.2f in %i frames" % (score, cnt))
            restart_delay = 60*2
        restart_delay -= 1
        if restart_delay==0:
            break

    return is_finish


if __name__ == "__main__":
    os.system('clear')    
    with open("left_situation.pkl", "rb") as f:
        #pickle.dump(, f)
        situation_for_calibration["LEFT"] = pickle.load(f)

    with open("up_situation.pkl", "rb") as f:
        situation_for_calibration["UP"] = pickle.load(f) 

    with open("right_situation.pkl","rb") as f:
        situation_for_calibration["RIGHT"] = pickle.load(f)
        
    with open("down_situation.pkl","rb") as f:
        situation_for_calibration["DOWN"] = pickle.load(f)

    cnt = 0
    for loop_cnt in range(5):
        
        os.system('clear') 
        print("キャリブレーションを実行します")
        time.sleep(1)
        for i in range(6):
            tmp_dir = dir[i%3]
            l = len(situation_for_calibration[tmp_dir])
            obs_num = random.randint(0, l-1)
            obs = situation_for_calibration[tmp_dir][obs_num]

            show_countdown()
            print("キャリブレーション中：", i+1, " / 6")
            calibration_and_check_input(env, obs)
            #calibration_by_image(env, obs)

        os.system('clear') 
        for i in Q_m.q_mapping:
            print(sum(Q_m.q_mapping[i]))
        
        print("ゲームを実行します．")
        time.sleep(1)
        show_countdown()
        print("ゲーム中")
        play_pong(cnt, loop_cnt)
        
        path = "with_calibration/"
        os.makedirs(path, exist_ok=True)
        
        with open(path +  "/" + "mapping.txt", "a") as tf:
            for i in range(4):
                s = 'CH'+str(i+1)+ " : " + dir[np.argmax(Q_m.q_mapping['CH'+str(i+1)])] + "\n"
                tf.write(s)
    date_label = datetime.datetime.now().strftime('%Y_%m_%d-%H%M%S')
    save_qmapping_log(Q_m, Q_m.log, "with_calibration/"+date_label+"/")

        