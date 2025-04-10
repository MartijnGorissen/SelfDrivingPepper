import numpy as np
import can
import struct
from typing import Optional, Dict, List, Literal, Tuple
import time

CAN_MSG_SENDING_SPEED = .040
instructie = [[0, 0.5, 0], [0, 0.6, 0], [0, 0.6, 0], [0, 0.7, 0], [0, 0.8, 0], [0, 0.9, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]
def initialize_can():
    """
    Set up the can bus interface
    """
    bus = can.Bus(interface='socketcan', channel='can0', bitrate=500000)
    return bus
    
def main(session):
    bus = initialize_can()
        
    try:
        # Define messages
        brake_msg = can.Message(arbitration_id=0x110, is_extended_id=False, data=[0, 0, 0, 0, 0, 0, 0, 0])
        brake_task = bus.send_periodic(brake_msg, CAN_MSG_SENDING_SPEED)
        steering_msg = can.Message(arbitration_id=0x220, is_extended_id=False, data=[0, 0, 0, 0, 0, 0, 0, 0])
        steering_task = bus.send_periodic(steering_msg, CAN_MSG_SENDING_SPEED)
        throttle_msg = can.Message(arbitration_id=0x330, is_extended_id=False, data=[0, 0, 0, 0, 0, 0, 0, 0])
        throttle_task = bus.send_periodic(throttle_msg, CAN_MSG_SENDING_SPEED)

        # Start running       
        try:
            for task_per_second in instructie:
                steering_angle, throttle, brake = task_per_second
    
                brake_msg.data = [int(99*max(0, brake))] + 7*[0]
                steering_msg.data = list(bytearray(struct.pack("f", float(steering_angle)))) + [0]*4
                throttle_msg.data = [int(99*max(0, throttle)), 0, 1] + 5*[0]
    
                brake_task.modify_data(brake_msg)
                steering_task.modify_data(steering_msg)
                throttle_task.modify_data(throttle_msg)
                time.sleep(2)
                
        except KeyboardInterrupt:
            pass

    finally:
        throttle_task.stop()
        steering_task.stop()
        brake_task.stop()
