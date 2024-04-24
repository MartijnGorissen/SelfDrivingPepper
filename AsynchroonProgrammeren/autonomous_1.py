import asyncio
import sys
import cv2
import can
import struct
import time
from typing import Dict, List
from ultralytics import YOLO
from LaneLines import LaneLines
import video as video

# Laden van stoplicht model
model = YOLO("traffic-lights.pt")

CAN_MSG_SENDING_SPEED = 0.040


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


def initialize_can():
    """
    Set up the can bus interface
    """
    bus = can.Bus(interface='socketcan', channel='can0', bitrate=500000)
    return bus


async def brake_or_drive_task(camera, brake_task, throttle_task):
    """
    Asynchrone task voor de stoplichten
    """
    while True:
        # Lezen van de camera
        _, frame = camera.read()

        # resultaten van het model ophalen
        results = model([frame])

        # Standaard waarden voor tijdens rijden
        # Pas throttle aan als je sneller wilt testen
        brake = 0
        throttle = 0.5

        if results[0].boxes:
            # Ophalen eerste box en waarden
            detection = results[0].boxes[0]
            detected_class = detection.cls
            confidence = detection.conf

            # Kijken voor rood licht
            if detected_class == 4 and confidence >= 0.4:
                # Wacht 1 seconde met remmen
                # Waarde bij sleep aanpassen voor meer delay of minder
                brake = 1
                throttle = 0

        # Aanpassen CAN berichten
        brake_msg = can.Message(
            arbitration_id=0x110,
            is_extended_id=False,
            data=[int(99 * max(0.5, brake)), 0, 0, 0, 0, 0, 0, 0]
        )

        throttle_msg = can.Message(
            arbitration_id=0x330,
            is_extended_id=False,
            data=[int(99 * throttle), 0, 1, 0, 0, 0, 0, 0]
        )

        # Sturen CAN berichten
        brake_task.modify_data(brake_msg)
        throttle_task.modify_data(throttle_msg)

        # Heel kort wachten voor andere tasks
        await asyncio.sleep(0.01)


async def lane_detection_task(camera, ll, steering_task):
    """
    Functie om asynchroon het sturen te regelen
    """
    while True:
        _, frame = camera.read()
        steering_angle, _, _ = ll.process_image_to_instructie(frame)

        # Aanpassen CAN bericht
        steering_msg = can.Message(
            arbitration_id=0x220,
            is_extended_id=False,
            data=list(
                bytearray(
                    struct.pack(
                        "f", float(steering_angle)
                            )
                        )
                    ) + [0]*4
        )

        # Sturen CAN bericht
        steering_task.modify_data(steering_msg)

        # Controleer hiermee de execution rate (momenteel elke 0.04 sec)
        await asyncio.sleep(0.04)


async def main():
    print("Initializing...")
    bus = initialize_can()
    cameras = initialize_cameras()
    front_camera = cameras["front"]

    ll = LaneLines()
    try:
        # CAN berichten aanmaken
        brake_msg = can.Message(
            arbitration_id=0x110,
            is_extended_id=False,
            data=[0, 0, 0, 0, 0, 0, 0, 0]
            )
        brake_task = bus.send_periodic(brake_msg, CAN_MSG_SENDING_SPEED)

        steering_msg = can.Message(
            arbitration_id=0x220,
            is_extended_id=False,
            data=[0, 0, 0, 0, 0, 0, 0, 0]
            )
        steering_task = bus.send_periodic(steering_msg, CAN_MSG_SENDING_SPEED)

        throttle_msg = can.Message(
            arbitration_id=0x330,
            is_extended_id=False,
            data=[0, 0, 0, 0, 0, 0, 0, 0]
            )
        throttle_task = bus.send_periodic(throttle_msg, CAN_MSG_SENDING_SPEED)

        print("Running...")
        start_time = time.time()
        frame_count = 0

        try:
            # De asynchrone taken aanmaken en uitvoeren
            tasks = [
                brake_or_drive_task(front_camera, brake_task, throttle_task),
                lane_detection_task(front_camera, ll, steering_task)
            ]
            await asyncio.gather(*tasks)

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
    asyncio.run(main())
