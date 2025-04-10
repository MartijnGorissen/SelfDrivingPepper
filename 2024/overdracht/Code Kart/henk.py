import numpy as np
import can
import struct
from typing import Optional, Dict, List, Literal, Tuple
from LaneLines import *
from time import sleep
import cv2
import time
import video as video
import sys
from ultralytics import YOLO
from PIL import Image
import easyocr
import asyncio
from rplidar import RPLidar


# Initieren van de nodige standaarden
CAN_MSG_SENDING_SPEED = .040
model = YOLO("obj_det_zon.pt")
parking_model = YOLO("park_detectie2.pt")
reader = easyocr.Reader(['en'])
ll = LaneLines()
lidar = RPLidar('/dev/ttyUSB0')
info = lidar.get_info()
health = lidar.get_health()
lidar.clean_input()


def refresh_lidar():
    """
    Deze code reset de lidar. Dit is nodig
    om te voorkomen dat de code stopt met werken
    """
    lidar.stop()
    lidar.start_motor()


def initialize_can() -> can.Bus:
    """
    Initialiseert de CAN-bus interface.

    Returns:
    - can.Bus: Geconfigureerde CAN-bus object.
    """
    # Initialiseer de CAN-bus
    bus = can.Bus(interface='socketcan', channel='can0', bitrate=500000)
    return bus


def initialize_cameras() -> Dict[str, cv2.VideoCapture]:
    """
    Initialiseert de OpenCV cameravastlegapparaten.

    Returns:
    - Dict[str, cv2.VideoCapture]: Een dictionary met
                                   de geconfigureerde camera's.
    """
    # Haal de cameracofiguratie op
    config: video.CamConfig = video.get_camera_config()

    if not config:
        # Geef een foutmelding als er geen geldige configuratie is gevonden
        print(
            'No valid video configuration found!',
            file=sys.stderr
            )

        # Stop het programma met foutcode 1
        exit(1)

    # Initialiseer een dictionary voor de camera's
    cameras: Dict[str, cv2.VideoCapture] = dict()

    for camera_type, path in config.items():
        # Maak een opname-object voor de camera
        capture = cv2.VideoCapture(path)

        # Stel de framebreedte in op 848 pixels
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 848)

        # Stel de framehoogte in op 480 pixels
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Schakel de autofocus uit
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        # Stel de focus in op oneindig
        capture.set(cv2.CAP_PROP_FOCUS, 0)

        # Stel de codec in op MJPG, LET OP VOOR CODEC BIJ 60FPS
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Stel de framesnelheid in op 30 FPS (kan naar 60)
        capture.set(cv2.CAP_PROP_FPS, 30)

        # Voeg de camera toe aan de dictionary
        cameras[camera_type] = capture

    return cameras


def predict_image(input, model):
    """
    Dit is de code die elke keer wordt gebruikt om voorspellingen te maken
    """
    img = input
    img_obb = img
    # Voorspellen met het Yolo model
    results = model.predict(img)
    # Een array maken van de image
    img = Image.fromarray(img_obb)
    return results, img


class BaseMode:
    """
    Basis modus voor objectherkenning en rij-instructies.
    Verwerkt de basisobjecten zoals stoplichten,
    snelheidslimieten, zebrapaden en auto's.

    Attributen:
    - model: Het objectherkenningsmodel.
    - max_throttle: De maximale gaspedaalpositie.
    - counter: Teller voor het bijhouden van de status.
    """
    def __init__(self, model, max_throttle, counter):
        self.model = model
        self.max_throttle = max_throttle
        self.counter = counter

    def handle(self, frame, mode):
        """
        Verwerkt een frame en bepaalt de rij-instructies op basis
        van gedetecteerde objecten.

        Parameters:
        - frame: Het huidige beeldframe.
        - mode: De huidige rijmodus.
        """
        self.results, self.img = predict_image(frame, self.model)

        for box in self.results[0].boxes:
            obj = box.cls.tolist()[0]

            if obj == 3:
                return self._redlight(box)

            elif obj == 4:
                return self._speedlimit(box)

            elif obj == 5:
                return self._zebra(box)

            elif obj == 0:
                return self._auto(box)

        return self.max_throttle, mode, None, None, None, 0

    def _bbox(self, box):
        """
        Berekent de afmetingen van de bounding box.

        Parameters:
        - box: De bounding box van het gedetecteerde object.
        """
        x1, y1, x2, y2 = box.xyxy[0]
        self.length = (y2 - y1).item()
        self.width = (x2 - x1).item()
        return x1, y1, x2, y2

    def _redlight(self, box):
        """
        Verwerkt een gedetecteerd stoplicht.

        Parameters:
        - box: De bounding box van het stoplicht.
        """
        _, _, _, _ = self._bbox(box)

        if self.length > 32:
            return self.max_throttle, "base", 0, 0, 1, 0
        elif self.length > 27:
            return self.max_throttle, "base", 0, 0, 0, 0

    def _speed_to_throttle(speed):
        """
        Verandert de afgelezen snelheid naar throttle
        op basis van percentages.

        Parameters:
        - speed: De afgelezen snelheid.
        """
        MAX_SPEED = 120

        # Percentage van throttle berekenen
        throttle = round((speed / MAX_SPEED), 2)

        # Throttle tussen max en min zetten
        return max(0.0, min(throttle, 1.0))

    def _speedlimit(self, box):
        """
        Verwerkt een gedetecteerd snelheidslimietbord en
        past de maximale throttle aan.

        Parameters:
        - box: De bounding box van het snelheidslimietbord.
        """
        x1, y1, x2, y2 = self._bbox(box)

        if self.width > 40:
            sign_array = np.array(
                self.img.crop((int(x1), int(y1), int(x2), int(y2)))
                )
            speed_limit = reader.readtext(sign_array)
            try:
                speed_hmu = int(speed_limit[0][1]) * 10
                self.max_throttle = self._speed_to_throttle(speed_hmu)
            except (ValueError, IndexError, TypeError):
                pass

    def _zebra(self, box):
        """
        Verwerkt een gedetecteerd zebrapad.

        Parameters:
        - box: De bounding box van het zebrapad.
        """
        _, _, _, _ = self._bbox(box)
        class_predictions = self.results[0].boxes.cls.tolist()

        if 2 in class_predictions:
            if self.length > 100:
                return self.max_throttle, "zebra", 0, 0, 1, 0
            elif self.length > 60:
                return self.max_throttle, "base", 0, round(self.max_throttle / 2, 2), 0, 0
        else:
            if self.length > 30:
                return self.max_throttle, "base", 0, None, None, 0

    def _auto(self, box):
        """
        Verwerkt een gedetecteerde auto.

        Parameters:
        - box: De bounding box van de auto.
        """
        _, _, _, _ = self._bbox(box)

        if self.length > 100:
            return self.max_throttle, "auto1", 0, 0, 1, 0


class ZebraMode:
    """
    Modus voor het verwerken van zebrapaden en gerelateerde handelingen.

    Attributen:
    - model: Het objectherkenningsmodel.
    - max_throttle: De maximale gaspedaalpositie.
    - counter: Teller voor het bijhouden van de status.
    """
    def __init__(self, model, max_throttle, counter):
        self.model = model
        self.max_throttle = max_throttle
        self.counter = counter

    def handle(self, frame, mode):
        """
        Verwerkt een frame en bepaalt de rij-instructies
        op basis van de huidige modus en gedetecteerde objecten.

        Parameters:
        - frame: Het huidige beeldframe.
        - mode: De huidige rijmodus.
        """
        self.results, self.img = predict_image(frame, self.model)

        if mode == "zebra":
            return self._zebrapad()
        elif mode == "oversteken_naar_links":
            return self._oversteek_links()
        elif mode == "oversteken_naar_rechts":
            return self._oversteek_rechts()

        return self.max_throttle, mode, None, None, None, self.counter

    def _bbox(self, box):
        """
        Berekent de afmetingen van de bounding box.

        Parameters:
        - box: De bounding box van het gedetecteerde object.
        """
        x1, y1, x2, y2 = box.xyxy[0]
        self.length = (y2 - y1).item()
        self.width = (x2 - x1).item()
        return x1, y1, x2, y2

    def _zebrapad(self):
        """
        Verwerkt een gedetecteerd zebrapad
        """
        for box in self.results[0].boxes:
            obj = box.cls.tolist()[0]
            x1, y1, x2, y2 = self._bbox(box)

            if obj == 2:
                if x2 > 450:
                    return self.max_throttle, "oversteken_naar_links", 0, 0, 1, 0
                elif x2 < 450:
                    return self.max_throttle, "oversteken_naar_rechts", 0, 0, 1, 0

        return self.max_throttle, "zebra", None, None, None, 0

    def _oversteek_links(self):
        """
        Verwerkt een oversteek naar links bij een zebrapad
        """
        class_predictions = self.results[0].boxes.cls.tolist()

        for box in self.results[0].boxes:
            obj = box.cls.tolist()[0]
            x1, y1, x2, y2 = self._bbox(box)
            if 2 in class_predictions:
                if obj == 2:
                    if x2 < 300:
                        throttle = round(self.max_throttle / 1.5, 2)
                        return self.max_throttle, "oversteken_naar_links", 0, throttle, 0, 0
                    else:
                        return self.max_throttle, "oversteken_naar_links", 0, 0, 1, 0
            else:
                return self.max_throttle, "base", 0, None, None, 0

    def _oversteek_rechts(self):
        """
        Verwerkt een oversteek naar rechts bij een zebrapad
        """
        class_predictions = self.results[0].boxes.cls.tolist()

        for box in self.results[0].boxes:
            obj = box.cls.tolist()[0]
            x1, y1, x2, y2 = self._bbox(box)
            if 2 in class_predictions:
                if obj == 2:
                    if x2 > 650:
                        throttle = round(self.max_throttle / 1.5, 2)
                        return self.max_throttle, "oversteken_naar_rechts", 0, throttle, 0, 0
                    else:
                        return self.max_throttle, "oversteken_naar_rechts", 0, 0, 1, 0
            else:
                return self.max_throttle, "base", 0, None, None, 0


class AutoMode:
    """
    Modus voor het verwerken van geautomatiseerde rij-instructies.

    Attributen:
    - model: Het objectherkenningsmodel.
    - max_throttle: De maximale gaspedaalpositie.
    - counter: Teller voor het bijhouden van de status.
    """
    def __init__(self, model, max_throttle, counter):
        self.model = model
        self.max_throttle = max_throttle
        self.counter = counter

    def handle(self, frame, mode):
        """
        Verwerkt een frame en bepaalt de rij-instructies op basis van de huidige modus.

        Parameters:
        - frame: Het huidige beeldframe.
        - mode: De huidige rijmodus.
        """
        if mode == "xauto1":
            return self._xauto1()
        elif mode == "xauto2":
            return self._xauto2()
        elif mode == "xauto3":
            return self._xauto3()
        elif mode == "xauto4":
            return self._xauto4()
        elif mode == "xauto5":
            return self._xauto5()
        return self.max_throttle, mode, None, None, None, 0

    def _xauto1(self):
        """
        Verwerkt de modus xauto1
        """
        if self.counter > 375:
            return self.max_throttle, "xauto2", 0, 0, 0, 0
        else:
            return self.max_throttle, "xauto1", -0.4, 0.5, 0, self.counter + 1

    def _xauto2(self):
        """
        Verwerkt de modus xauto2
        """
        if self.counter > 200:
            return self.max_throttle, "xauto3", 0, 0, 0, 0
        else:
            return self.max_throttle, "xauto2", 0.9, 0.5, 0, self.counter + 1

    def _xauto3(self):
        """
        Verwerkt de modus xauto3
        """
        refresh_lidar()
        for i, scan in enumerate(lidar.iter_scans()):
            distances = [d for q, a, d in scan if (d < 3000) and (180 < a < 280)]
            if len(distances) == 0:
                return self.max_throttle, "xauto4", 0, 0, 0, 0
            else:
                return self.max_throttle, "xauto3", 0, 0.5, 0, 0

    def _xauto4(self):
        """
        Verwerkt de modus xauto4
        """
        if self.counter > 500:
            return self.max_throttle, "xbocht1", 0, 0, 0, 0
        else:
            return self.max_throttle, "xauto4", 0.3, 0.4, 0, self.counter + 1


class BochtMode:
    """
    Modus voor het verwerken van de grote bocht.

    Attributen:
    - model: Het objectherkenningsmodel.
    - max_throttle: De maximale gaspedaalpositie.
    - counter: Teller voor het bijhouden van de status.
    """
    def __init__(self, model, max_throttle, counter):
        self.model = model
        self.max_throttle = max_throttle
        self.counter = counter

    def handle(self, frame, mode):
        """
        Verwerkt een frame en bepaalt de rij-instructies op basis van de huidige modus.

        Parameters:
        - frame: Het huidige beeldframe.
        - mode: De huidige rijmodus.
        """
        if mode == "xbocht1":
            return self._xbocht1()
        elif mode == "xbocht2":
            return self._xbocht2()
        return self.max_throttle, mode, None, None, None, 0

    def _xbocht1(self):
        """
        Verwerkt de modus xbocht1
        """
        if self.counter > 600:
            return self.max_throttle, "xbocht2", 0, 0, 0, 0
        else:
            return self.max_throttle, "xbocht1", 0, 0.8, 0, self.counter + 1

    def _xbocht2(self):
        """
        Verwerkt de modus xbocht2
        """
        if self.counter > 350:
            return self.max_throttle, "xpark0", 0, 0, 0, 0
        else:
            return self.max_throttle, "xbocht2", -1.0, 0.8, 0, self.counter + 1


class ParkMode:
    """
    Modus voor het verwerken van het inparkeren.

    Attributen:
    - model: Het objectherkenningsmodel.
    - max_throttle: De maximale gaspedaalpositie.
    - counter: Teller voor het bijhouden van de status.
    """
    def __init__(self, model, max_throttle, counter):
        self.model = model
        self.max_throttle = max_throttle
        self.counter = counter

    def handle(self, frame, mode):
        """
        Verwerkt een frame en bepaalt de rij-instructies op basis van de huidige modus.

        Parameters:
        - frame: Het huidige beeldframe.
        - mode: De huidige rijmodus.
        """
        if mode == "xpark0":
            return self._xpark0()
        elif mode == "xpark1":
            return self._xpark1()
        elif mode == "xpark2":
            return self._xpark2()
        elif mode == "xpark3":
            return self._xpark3()
        elif mode == "xpark4":
            return self._xpark4()
        elif mode == "xpark5":
            return self._xpark5()
        elif mode == "STOP":
            return self._stop()
        return self.max_throttle, mode, None, None, None, 0

    def _xpark0(self):
        """
        Verwerkt de modus xpark0
        """
        refresh_lidar()

        for i, scan in enumerate(lidar.iter_scans()):
            if i == 0:
                pass
            else:
                distances = [d for q, a, d in scan if (d < 3000) and (267 < a < 273)]

                if len(distances) > 0:
                    return self.max_throttle, "xpark1", 0, 0, 0, 0
                else:
                    return self.max_throttle, "xpark0", None, 0.5, 0, 0

    def _xpark1(self):
        """
        Verwerkt de modus xpark1
        """
        refresh_lidar()

        for i, scan in enumerate(lidar.iter_scans()):
            if i == 0:
                pass
            else:
                distances = [d for q, a, d in scan if (d < 3000) and (267 < a < 273)]

                if len(distances) == 0:
                    return self.max_throttle, "xpark2", 0, 0, 0, 0
                else:
                    return self.max_throttle, "xpark1", None, 0.5, 0, 0

    def _xpark2(self):
        """
        Verwerkt de modus xpark2
        """
        refresh_lidar()

        for i, scan in enumerate(lidar.iter_scans()):
            if i == 0:
                pass
            else:
                distances = [d for q, a, d in scan if (d < 3000) and (267 < a < 273)]

                if len(distances) > 0:
                    return self.max_throttle, "xpark3", 0, 0, 0, 0
                else:
                    return self.max_throttle, "xpark2", None, 0.5, 0, 0

    def _xpark3(self):
        """
        Verwerkt de modus xpark3
        """
        if self.counter > 400:
            return self.max_throttle, "xpark4", 0, 0, 0, 0
        else:
            return self.max_throttle, "xpark3", -0.4, -0.2, 0, self.counter + 1

    def _xpark4(self):
        """
        Verwerkt de modus xpark4
        """
        if self.counter > 300:
            return self.max_throttle, "xpark5", 0, 0, 0, 0
        else:
            return self.max_throttle, "xpark4", 0.6, -0.2, 0, self.counter + 1

    def _xpark5(self):
        """
        Verwerkt de modus xpark5
        """
        if self.counter > 300:
            return self.max_throttle, "STOP", 0, 0, 0, 0
        else:
            return self.max_throttle, "xpark5", 0.2, 0.2, 0, self.counter + 1

    def _stop(self):
        """
        Verwerkt de STOP-modus
        """
        return self.max_throttle, "STOP", 0, 0, 1, 0


def obj_drive(frame, max_throttle, mode, counter):
    """
    Deze functie behandeld alle modi in een compact if statement

    Parameters:
    - frame: Het huidige beeldframe.
    - max_throttle: De maximale gaspedaalpositie.
    - mode: De huidige rijmodus.
    - counter: Teller voor het bijhouden van de status.

    Returns:
    - De volgende variabelen, altijd in dezeflde volgorde:
        - max_throttle
        - mode
        - steering_angle
        - throttle
        - brake
        - counter
    """
    # If statement om de juist modus te selecteren
    # en de class van die modus te initialiseren
    if mode == "zebra" or mode.startswith("oversteken"):
        instructies = ZebraMode(model, max_throttle, counter)
    elif mode.startswith("xauto"):
        instructies = AutoMode(model, max_throttle, counter)
    elif mode.startswith("xbocht"):
        instructies = BochtMode(model, max_throttle, counter)
    elif mode.startswith("xpark") or mode == "STOP":
        instructies = ParkMode(parking_model, max_throttle, counter)
    else:
        instructies = BaseMode(model, max_throttle, counter)

    # Het gebruiken van de handle methode die in elke class
    # de taken kan afhandelen
    return instructies.handle(frame, mode)


async def process_image(frame):
    """
    Conversie van synchroon naar asynchroon
    """
    return ll.process_image_to_instructie(frame)


async def async_drive(frame, max_throttle_, mode_, counter_):
    """
    Conversie van synchroon naar asynchroon
    """
    return obj_drive(frame, max_throttle_, mode_, counter_)


async def main():
    """
    Aansturen van de kart
    """
    # Initialiseren van de CAN-bus en de camera's plus wat basis parameters
    print("Initializing....")
    bus = initialize_can()

    cameras = initialize_cameras()
    front_camera = cameras["front"]

    max_throttle_ = 1
    counter_ = 0
    mode_ = 'base'
    obj_drive_task = None

    try:
        print("Booting.....")
        # Aanmaken van de CAN-bus berichten en de taken
        brake_msg = can.Message(arbitration_id=0x110, is_extended_id=False, data=[0, 0, 0, 0, 0, 0, 0, 0])
        brake_task = bus.send_periodic(brake_msg, CAN_MSG_SENDING_SPEED)
        steering_msg = can.Message(arbitration_id=0x220, is_extended_id=False, data=[0, 0, 0, 0, 0, 0, 0, 0])
        steering_task = bus.send_periodic(steering_msg, CAN_MSG_SENDING_SPEED)
        throttle_msg = can.Message(arbitration_id=0x330, is_extended_id=False, data=[0, 0, 0, 0, 0, 0, 0, 0])
        throttle_task = bus.send_periodic(throttle_msg, CAN_MSG_SENDING_SPEED)

        print("Running.....")
        # Beginnen met de while-loop
        start_time = time.time()
        frame_count = 0
        try:
            while True:
                # Ophalen huidig frame
                _, frame = front_camera.read()

                # Kijken of er al een OBJ detectie taak bezig is, zo ja dan gaat de code door
                if obj_drive_task is None or obj_drive_task.done():
                    obj_drive_task = asyncio.create_task(async_drive(frame, max_throttle_, mode_, counter_))

                # Uitvoeren lane detection
                process_image_task = asyncio.create_task(process_image(frame))

                # Asyncio gebruiken om de code te runnen en de taken uit te voeren
                results = await asyncio.gather(obj_drive_task, process_image_task)
                (max_throttle, mode, steering_angle, throttle, brake, counter), (ll_steering_angle, ll_throttle, ll_brake, _) = results

                # Ik weet niet waarom maar anders werkt de code gewoon niet. Dus dit moet
                # Het ziet er inderdaad nutteloos uit, maar het is nodig!!
                max_throttle, mode, steering_angle, throttle, brake, counter = (max_throttle, mode, steering_angle, throttle, brake, counter)
                ll_steering_angle, ll_throttle, ll_brake, _ = (ll_steering_angle, ll_throttle, ll_brake, _)

                # Veranderen van de mode, max_throttle en counter voor volgende iteratie
                mode_ = mode
                max_throttle_ = max_throttle
                counter_ = counter

                # Lane Detection gebruiken als er geen stuur input is van OBJ
                if steering_angle is None:
                    steering_angle = ll_steering_angle

                # Voluit laten rijden als er geen gas input is van OBJ
                if throttle is None:
                    throttle = max_throttle

                # Rem op nul zetten als er geen rem input is
                if brake is None:
                    brake = ll_brake

                # Throttle halveren zodat de kart traag genoeg gaat
                throttle = round((throttle / 2), 2)

                # Updaten van de canbus berichten
                brake_msg.data = [int(99*max(0, brake))] + 7*[0]
                steering_msg.data = list(
                    bytearray(
                        struct.pack("f", float(steering_angle))
                        )
                    ) + [0]*4
                
                if throttle < 0:
                    throttle * -1
                    throttle_msg.data = [int(99*max(0, throttle)), 0, 2] + 5*[0]
                else:
                    throttle_msg.data = [int(99*max(0, throttle)), 0, 1] + 5*[0]

                # Versturen van de updates naar de canbus
                brake_task.modify_data(brake_msg)
                steering_task.modify_data(steering_msg)
                throttle_task.modify_data(throttle_msg)

                # Teller voor het aantal frames waarop is voorspeld
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
        lidar.stop()

if __name__ == '__main__':
    asyncio.run(main())
