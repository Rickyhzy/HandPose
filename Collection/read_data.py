import serial
import numpy as np
import pandas as pd
import time

'''
手动批量记录传感器值，需要修改name
'''
DB = []
SERIAL_PORT = 'com10'
BAUD_RATE = 115200
TIMEOUT = 10

def init():
    ser = serial.Serial(port=SERIAL_PORT,baudrate=BAUD_RATE,timeout=TIMEOUT)
    if(ser.is_open):
        print("ser is ok")
    return ser


def to_file(data,name):
    file = open('../data/deal/{}.txt'.format(name),'w')
    for i in range(len(data)):
        s = str(data[i]).replace("'", '').replace("[",'').replace("]",'').replace(',','') + '\n'
        file.write(s)
    file.close()

def arr2csv(data,name):
    data = pd.DataFrame(data)
    data.to_csv('../data/private/{}.csv'.format(name),index_label=None,header=None,index=None)


if __name__ == '__main__':
    flag = True
    ser = init()
    ser.flushInput()
    start_time = time.time()
    while True:
        try:
            if flag:
                ser.flushInput()
                flag = False
            data = ser.readline().strip().decode('UTF-8').replace('\n', '').split(',')
            print(data,type(data))
            DB.append(data)
            over_time = time.time()
            time.sleep(0.01)
            if (over_time - start_time) > 5:
                print('data is SAMPLED')
                break
        except serial.SerialException:
            print('Data could not be read')
    ser.close()
    # to_file(DB,'one')
    arr2csv(DB,'OK')
