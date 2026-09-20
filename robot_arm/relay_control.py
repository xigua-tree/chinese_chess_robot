import serial
import time

ser = serial.Serial("COM9", 115200)

while True:
    print("开")
    ser.setRTS(False)  # EN = 0（复位）
    ser.setDTR(False)  # BOOT = 0
    time.sleep(3)

    print("关")
    ser.setRTS(True)
    ser.setDTR(True)
    time.sleep(3)