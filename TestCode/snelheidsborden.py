import numpy as np
import can
import struct
from typing import Optional, Dict, List, Literal, Tuple
from LaneLines import *
from time import sleep
import cv2
import time
from ultralytics import YOLO
import video as video
import sys
import easyocr


reader = easyocr.Reader(['en'])
CAN_MSG_SENDING_SPEED = .040
MAX_SPEED = 120
class_naam = "TrafficSign"
# model = YOLO("Model_Name.pt")


def get_bbox(results, class_name):
    """
    Kijkt voor een gegeven class of er bounding
    boxes zijn gevonden.
    """
    # Kijken of alle items correct zijn
    for result in results:
        if hasattr(result, 'names') and hasattr(result, 'boxes'):
            # Zoeken van de class index
            class_index = None
            for index, name in result.names.items():
                if name == class_name:
                    class_index = index
                    break

            # Ga door als de class_index niet is gevonden
            if class_index is None:
                continue

            # Ophalven van de box voor de class
            for box in result.boxes:
                if box.cls == class_index:
                    # Ophalen van de box coordinaten
                    x1, y1, x2, y2 = box.xyxy[0]
                    width = x2 - x1
                    height = y2 - y1
                    return [x1, y1, width, height]

    # Return None als er geen box van die class is gevonden
    return None


def crop_bbox(image, bbox):
    """
    Snijd de bounding box uit de afbeelding en returnd
    de gesneden afbeelding.

    Parameters:
    - image: De originele afbeelding
    - bbox: De bounding box in de volgende vorm: [x, y, width, height].

    Returns:
    - bbox_img: De gesneden afbeelding
    """
    # Uitpakken bbox
    x, y, width, height = bbox

    # Omzetten van tensors naar int
    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)

    # Berekenen van de juiste coordinaten
    right = x + width
    bottom = y + height

    # Snijden van afbeelding
    bbox_img = image.crop((x, y, right, bottom))

    return bbox_img


def read_speedsign(sign):
    """
    Leest een speedsign af
    """
    # Omzetten van afbeelding naar np.array
    sign_array = np.array(sign)

    # Lezen van de np.array met EasyOCR
    speed_limit = reader.readtext(sign_array)

    # Ophalen van de snelheid en omzetten naar hm/u
    speed_hmu = speed_limit[0][1] * 10
    return speed_hmu


def handle_speedsign(image):
    """
    Behandeld het aflezen van speedsigns door
    middel van EasyOCR en Yolo
    """
    # Ophalen van de voorspelling
    results = model([image])

    # Ophalen van de Bounding Box
    bbox = get_bbox(results, class_naam)

    # Snijden van de box naar afbeelding
    sign = crop_bbox(image, bbox)

    # Aflezen van de afbeelding
    speed = read_speedsign(sign)

    # Omzetten van snelheidslimiet naar throttle
    # dmv maximum snelheid / snelheid voor % throttle
    throttle = MAX_SPEED / speed
    return throttle


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

        print("Running.....")
        # Start running
        start_time = time.time()
        frame_count = 0  
        try:
            

            temp = True
            while temp == True:
                _, frame = front_camera.read()
                temp = brake_or_drive(frame)

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
