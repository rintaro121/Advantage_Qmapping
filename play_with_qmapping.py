import argparse
import datetime
import os

import gym.wrappers
import numpy as np
from numpy.core.records import record
from numpy.lib.npyio import load
import pickle
from PIL import Image
import roboschool

from wrappers import action_space_discretizer
from q_mapping import Qmapping
from utils.utils import read_sensor_ch, read_CH_and_pressure
from save_log import save_qmapping_log

gif = []

parser = argparse.ArgumentParser()
parser.add_argument('dir', help="direction which boards move", choices=['vertical', "diagonal"])
args = parser.parse_args()

CHs = ["CH1", "CH2", "CH3", "CH4"]
#ADDRESS = "/dev/tty.usbmodem14301"
ADDRESS = '/dev/tty.usbmodem14401'
#Q_PATH = "weight/oblique_direction.dat"

dir = ["LEFT", "UP", "RIGHT", "DOWN"]
DEFAULT_ENV_NAME = "RoboschoolPong-v1"  # Use a longer version of Pong for demonstration (needs to be defined in source)
MAKE_VIDEO = True  # Set true or false here to record video OR render, not both



def main():
    env = gym.make(DEFAULT_ENV_NAME)
    env = action_space_discretizer(env, 2, args.dir)

    

    env.reset()
    still_open = True

    if args.dir == "vertical":
        Q_m = Qmapping("weight/PPO_RoboschoolPong-v1_0_0.pth", args)
    elif args.dir == "diagonal":
        Q_m = Qmapping("weight/oblique_direction.dat", args)

    #recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env, "./recording.mp4", enabled=MAKE_VIDEO)
    for i in range(5):
        frame = 0
        score = 0
        restart_delay = 0

        obs = env.reset()

        mycam = env.unwrapped.scene.cpp_world.new_camera_free_float(320,200,"my_camera")
        mycam.move_and_look_at( 0, 0, 3, 0, 0, 0)

        while True:

            #action_id = read_sensor_ch(ADDRESS)

            operation_ch, puressure_info = read_CH_and_pressure(ADDRESS)

            action = []

            done = False
            if not operation_ch:
                action = -1
            else:
                #action = Q_m.interpret(action_id, obs)
                action = Q_m.reversable_interpret(operation_ch,  obs, calibration=False, frame=frame, play_only=True)

                ### log input and returned action
                Q_m.log.push_cnt_update(operation_ch)
                Q_m.log.move_cnt_update(action)

            Q_m.log.puressure_log_update(puressure_info)
                

            obs, r, done, _ = env.step(np.array(action))

            Q_m.log.observation_log_update(obs)

            score += r
            frame += 1

    
            still_open = env.render("human")

            rgb, _, _, _, _ = mycam.render(False, False, False)
            mycam.test_window() 

            if not done: continue
            if restart_delay==0:
                print("score=%0.2f in %i frames" % (score, frame))
            #    if still_open!=True:      # not True in multiplayer or non-Roboschool environment
            #        break
                restart_delay = 60*2  # 2 sec at 60 fps
            restart_delay -= 1
            if restart_delay==0: break


        path = "play_with_qmapping/"
        os.makedirs(path,  exist_ok=True)
        with open(path + "/" + "mapping.txt", "a") as tf:
            for i in range(4):
                s = 'CH'+str(i+1)+ " : " + dir[np.argmax(Q_m.q_mapping['CH'+str(i+1)])] + "\n"
                tf.write(s)
    date_label = datetime.datetime.now().strftime('%Y_%m_%d-%H%M%S')
    save_qmapping_log(Q_m, Q_m.log, "play_with_qmapping/"+date_label+"/")

if __name__ == "__main__":
    main()