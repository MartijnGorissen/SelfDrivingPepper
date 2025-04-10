import numpy as np
import can
import struct
from typing import Optional, Dict, List, Literal, Tuple
from LaneLines import *
from time import sleep
import cv2
import time
from ultralytics import YOLO

model = YOLO("traffic-lights.pt")  # load a pretrained model (recommended for training)
def brake_or_drive(image):
    results = model([image])
    brake = results[0].boxes.cls[0] == 4 and results[0].boxes.conf[0] >= 0.4
    return brake

CAN_MSG_SENDING_SPEED = .040

def initialize_cameras() -> Dict[str, cv2.VideoCapture]:
    """
    Initialize the opencv camera capture devices.
    """
    config: video.CamConfig = video.get_camera_config()
    if not config:
        print('No valid video configuration found!', file=sys.stderr)
        exit(1)
    cameras: Dict[str, cv2.VideoCapture] = dict()
    for camera_type, path in config.items():
        capture = cv2.VideoCapture(path)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        capture.set(cv2.CAP_PROP_FOCUS, 0)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #important to set right codec to enable 60fps
        capture.set(cv2.CAP_PROP_FPS, 30) #make 60 to enable 60FPS
        cameras[camera_type] = capture
    return cameras

def initialize_can():
    """
    Set up the can bus interface
    """
    bus = can.Bus(interface='socketcan', channel='can0', bitrate=500000)
    return bus
    
def main():
    print("Initializing....")
    bus = initialize_can()
    
    cameras = initialize_cameras()
    front_camera = cameras["front"]
        
    try:
        print("Booting.....")
        # Define messages
        brake_msg = can.Message(arbitration_id=0x110, is_extended_id=False, data=[0, 0, 0, 0, 0, 0, 0, 0])
        brake_task = bus.send_periodic(brake_msg, CAN_MSG_SENDING_SPEED)
        steering_msg = can.Message(arbitration_id=0x220, is_extended_id=False, data=[0, 0, 0, 0, 0, 0, 0, 0])
        steering_task = bus.send_periodic(steering_msg, CAN_MSG_SENDING_SPEED)
        throttle_msg = can.Message(arbitration_id=0x330, is_extended_id=False, data=[0, 0, 0, 0, 0, 0, 0, 0])
        throttle_task = bus.send_periodic(throttle_msg, CAN_MSG_SENDING_SPEED)
        ll = LaneLines()

        print("Running.....")
        # Start running
        start_time = time.time()
        frame_count = 0  
        try:
            temp = True
            while temp == True:
                temp = brake_or_drive(frame)

            while True:
                _, frame = front_camera.read()
                
                cv2.imshow('Camera preview', frame)
                cv2.waitKey(1)

                steering_angle, throttle, brake = ll.process_image_to_instructie(frame)
    
                brake_msg.data = [int(99*max(0, brake))] + 7*[0]
                steering_msg.data = list(bytearray(struct.pack("f", float(steering_angle)))) + [0]*4
                throttle_msg.data = [int(99*max(0, throttle)), 0, 1] + 5*[0]
    
                brake_task.modify_data(brake_msg)
                steering_task.modify_data(steering_msg)
                throttle_task.modify_data(throttle_msg)

                frame_count += 1
        except KeyboardInterrupt:
            pass

        end_time = time.time()
        time_diff = end_time - start_time

        print(f'Time elapsed: {time_diff:.2f}s')
        print(f'Frames processed: {frame_count}')
        print(f'FPS: {frame_count/time_diff:.2f}')

    finally:
        print("Stopping.....")
        throttle_task.stop()
        steering_task.stop()
        brake_task.stop()

if __name__ == '__main__':
    main()
