import numpy as np
import serial

def read_sensor_ch(adr):
    #time.sleep(0.0001)
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

    if sum(data_int) >= 30:
        return CHs[np.argmax(data_int)]
        #return dir[np.argmax(data_int)]
    else:
        return None

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