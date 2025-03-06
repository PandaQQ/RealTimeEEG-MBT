import serial
import time
import binascii
from threading import Lock

COM_PORT = 'COM3'

class SerialPort:
    _instance = None
    _lock = Lock()

    def __new__(cls, port, baud):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SerialPort, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, port, baud):
        if self._initialized:
            return
        self.port = serial.Serial(port, baud)
        self.port.close()
        if not self.port.isOpen():
            self.port.open()
        self._initialized = True

    def port_open(self):
        if not self.port.isOpen():
            self.port.open()

    def port_close(self):
        self.port.close()

    def send_data(self, data):
        self.port.write(data)

def show_red():
    serialPort = COM_PORT
    baudRate = 9600
    RED = bytes.fromhex('11')
    mSerial = SerialPort(serialPort, baudRate)
    mSerial.port_open()
    mSerial.send_data(RED)

def show_green():
    serialPort = COM_PORT
    baudRate = 9600
    GREEN = bytes.fromhex('14')
    mSerial = SerialPort(serialPort, baudRate)
    mSerial.port_open()
    mSerial.send_data(GREEN)

def close():
    serialPort = COM_PORT
    baudRate = 9600
    CLOSERED = bytes.fromhex('21')
    CLOSEYE = bytes.fromhex('22')
    CLOSEGR = bytes.fromhex('24')
    CLOSEBU = bytes.fromhex('28')
    mSerial = SerialPort(serialPort, baudRate)
    mSerial.port_open()
    mSerial.send_data(CLOSERED)
    mSerial.send_data(CLOSEYE)
    mSerial.send_data(CLOSEGR)
    mSerial.send_data(CLOSEBU)


def red_control():
    close()
    show_red()

def green_control():
    close()
    show_green()

if __name__ == '__main__':
    red_control()
    time.sleep(1)
    green_control()
    time.sleep(1)
    close()