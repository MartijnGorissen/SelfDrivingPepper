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
class_naam = "snelheid"
model = YOLO("best.pt")


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
    speed_hmu = int(speed_limit[0][1]) * 10
    return speed_hmu


def speed_to_throttle(speed):
    """
    Veranderd de afgelezen snelheid naar throttle
    op basis van percentages
    """
    # Percentage van throttle berekenen
    throttle = round((speed / MAX_SPEED), 2)

    # Throttle tussen max en min zetten
    # Indien de throttle te hoog zou zijn
    if throttle > 1.0:
        throttle = 1.0
    elif throttle < 0.0:
        throttle = 0.0
    else:
        throttle = throttle

    return throttle


def handle_speedsign(image):
    """
    Behandeld het aflezen van speedsigns door
    middel van EasyOCR en Yolo
    """
    # Ophalen van de voorspelling
    results = model([image])

    # Ophalen van de Bounding Box
    bbox = get_bbox(results, class_naam)

    if bbox is not None:
        try:
            # Snijden van de box naar afbeelding
            sign = crop_bbox(image, bbox)

            # Aflezen van de afbeelding
            speed = read_speedsign(sign)

            # Omzetten van snelheidslimiet naar throttle
            throttle = speed_to_throttle(speed)
            return throttle, speed
        except:
            pass

    else:
        throttle = 1.0
        speed = MAX_SPEED
        return throttle, speed


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
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        capture.set(cv2.CAP_PROP_FOCUS, 0)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_FPS, 30)
        cameras[camera_type] = capture
    return cameras


def main():
    print("Initializing....")

    cameras = initialize_cameras()
    front_camera = cameras["front"]

    try:
        print("Booting.....")
        # Start running
        start_time = time.time()
        frame_count = 0

        print("Running.....")
        try:
            while True:
                _, frame = front_camera.read()
                throttle, speed = handle_speedsign(frame)
                print(f"Throttle is {throttle} by speedsign {speed}")
                # sleep(0.5)

        except KeyboardInterrupt:
            pass

        end_time = time.time()
        time_diff = end_time - start_time

        print(f'Time elapsed: {time_diff:.2f}s')
        print(f'Frames processed: {frame_count}')
        print(f'FPS: {frame_count/time_diff:.2f}')

    finally:
        print("Stopping.....")


if __name__ == '__main__':
    main()
