import gym, roboschool, sys
import numpy as np
import os
import datetime

import serial
import numpy as np
import time

import pickle

#ADDRESS = "/dev/tty.usbmodem144401"
ADDRESS = '/dev/tty.usbmodem14401'


def read_CH_and_pressure(adr):
    CHs = ["CH1","CH4","CH2","CH3"]
    dir = [0,3,2,1]

    ser = serial.Serial(adr,115200)
    ser.write(b'')
    ser.write(b'020201\n')
    res = ser.readline()
    res = res.decode("utf-8")
    data = []
    for i in range(5):
        data.append(res[i*4:i*4+4])
    data = data[1:]
    data_int = []
    for i in range(4):
        data_int.append(int( data[i],16 ))

    ch = None
    info = None

    if sum(data_int) >= 30:
        ch =  CHs[np.argmax(data_int)]
        info = data_int
    
    return ch, info

def read_sensor_value(adr):
    CHs = ["CH1","CH4","CH2","CH3"]
    dir = [0,3,2,1]

    ser = serial.Serial(adr,115200)
    ser.write(b'')
    ser.write(b'020201\n')
    res = ser.readline()
    res = res.decode("utf-8")
    data = []
    for i in range(5):
        data.append(res[i*4:i*4+4])
    data = data[1:]
    data_int = []
    for i in range(4):
        data_int.append(int( data[i],16 ))

    ret_value = [data_int[0],data_int[2],data_int[3],data_int[1]]
    return ret_value

def sensor_to_action(data):
    x = data[0] - data[2]
    y = data[1] - data[3]

    return [y * 0.1, x * 0.1]

def relu(x):
    return np.maximum(x, 0)

class SmallReactivePolicy:
    "Simple multi-layer perceptron policy, no internal state"
    def __init__(self, ob_space, ac_space):
        assert weights_dense1_w.shape == (ob_space.shape[0], 64)
        assert weights_dense2_w.shape == (64, 32)
        assert weights_final_w.shape  == (32, ac_space.shape[0])

    def act(self, ob):
        x = ob
        x = relu(np.dot(x, weights_dense1_w) + weights_dense1_b)
        x = relu(np.dot(x, weights_dense2_w) + weights_dense2_b)
        x = np.dot(x, weights_final_w) + weights_final_b
        return x

obs_log = []
puressure_log = []

def demo_run():
    env = gym.make("RoboschoolPong-v1")
    if len(sys.argv)==3: env.unwrapped.multiplayer(env, sys.argv[1], player_n=int(sys.argv[2]))

    pi = SmallReactivePolicy(env.observation_space, env.action_space)

    for i in range(1):
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()


        mycam = env.unwrapped.scene.cpp_world.new_camera_free_float(320,200,"my_camera")

        ### (camera_x, camera_y, camera_z, look_at_)
        mycam.move_and_look_at( 0, 0, 3, 0, 0, 0)

        while 1:
            #a = pi.act(obs)
            value = read_sensor_value(ADDRESS)
            puressure_log.append(value)
            action = sensor_to_action(value)
            
            #print(action)
            obs, r, done, _ = env.step(np.array(action))
            score += r
            frame += 1

            obs_log.append(obs)

            still_open = env.render("human")

            rgb, _, _, _, _ = mycam.render(False, False, False)
            mycam.test_window()
            #print(still_open)
            #if still_open==False:
            #    return
            if not done: continue
            if restart_delay==0:
                print("score=%0.2f in %i frames" % (score, frame))
            #    if still_open!=True:      # not True in multiplayer or non-Roboschool environment
            #        break
                restart_delay = 60*2  # 2 sec at 60 fps
            restart_delay -= 1
            if restart_delay==0: break
    date_label = datetime.datetime.now().strftime('%Y_%m_%d-%H%M%S')
    
    path = "default/" + date_label + "_" + "vertical" + "/"

    os.makedirs(path, exist_ok=True)
    with open(path+"normal_play_obs.pkl","wb") as f:
        pickle.dump(obs_log, f)

    with open(path+"normal_play_puressure.pkl","wb") as f:
        pickle.dump(puressure_log, f)
    
if __name__=="__main__":
    demo_run()
