import numpy as np
import can
import struct
from typing import Optional, Dict, List, Literal, Tuple
from LaneLines import *
from time import sleep
import cv2
import time
import video
import sys
import os
import threading
import time
from datetime import datetime
from collections import namedtuple
from queue import Queue


CAN_MSG_SENDING_SPEED = .040


class CanListener:
    """
    Een CAN Listener die luistert naar specifieke
    berichten en hun laatste waarden opslaat.
    """
    # Conversie dict voor de sensoren
    _id_conversion = {
        0x110: 'brake',               # Rem
        0x220: 'steering',            # Stuur
        0x330: 'throttle',            # Gas
        0x1e5: 'steering_sensor',     # Stuurhoek
        0x440: 'speed_sensor'         # Snelheid
    }

    def __init__(self, bus: can.Bus):
        """
        Initialisatie van de CanListener.

        Parameters:
        - bus (can.Bus): CAN-bus object waarop berichten worden ontvangen.
        """
        # Wijs het CAN-bus object toe
        self.bus = bus

        # Creëer thread voor het luisteren naar berichten
        self.thread = threading.Thread(
            target=self._listen, args=(), daemon=True
            )

        # Zet de luisterstatus uit
        self.running = False

        # Dict voor het opslaan van laatste sensorwaarden
        self.data: Dict[str, List[int]] = {
            name: None for name in self._id_conversion.values()
            }

    def start_listening(self):
        """
        Start het luisteren naar CAN-berichten.
        """
        # Begin met luisteren
        self.running = True

        # Start luisterthread
        self.thread.start()

    def stop_listening(self):
        """
        Stop het luisteren naar CAN-berichten.
        """
        # Stoppen met luisteren
        self.running = False

    def get_new_values(self):
        """
        Haal de nieuwste waarden op die zijn opgeslagen.
        """
        # Haal opgeslagen sensorwaarden op
        values = self.data
        return values

    def _listen(self):
        """
        Luistert continu naar CAN-berichten en slaat de waarde op
        als het bericht een bekend ID heeft.
        """
        # Blijf luisteren zolang luisterstatus True is
        while self.running:
            # Wacht op bericht met timeout van 0.5 seconden
            message: Optional[can.Message] = self.bus.recv(.5)
            if message and message.arbitration_id in self._id_conversion:
                # Als bericht ontvangen en ID bekend is:
                # Sla ontvangen data op bij betreffende sensor
                self.data[
                    self._id_conversion[message.arbitration_id]
                    ] = message.data


class ImageWorker:
    """
    Een werker die afbeeldingen naar de schijf schrijft.
    """

    def __init__(self, image_queue: Queue, folder: str):
        """
        Initialisatie van de ImageWorker.

        Parameters:
        - image_queue (Queue): Berichtenwachtrij voor afbeeldingsgegevens.
        - folder (str): Doelmap waar afbeeldingen worden opgeslagen.
        """
        # Object maken voor wachtrij
        self.queue = image_queue

        # Thread voor het verwerken van afbeeldingen
        self.thread = threading.Thread(
            target=self._process, args=(), daemon=True
            )

        # Object maken voor doelmap afbeeldingen
        self.folder: str = folder

    def start(self):
        """
        Start de verwerking van afbeeldingen.
        """
        # Start de verwerkingsthread
        self.thread.start()

    def stop(self):
        """
        Wacht tot alle afbeeldingen zijn verwerkt.
        """
        # Wacht tot alle taken in de wachtrij zijn voltooid
        self.queue.join()

    def put(self, data):
        """
        Voegt afbeeldingsgegevens toe aan de wachtrij.

        Parameters:
        - data: Afbeeldingsgegevens die moeten worden verwerkt.
        """
        # Voeg afbeeldingsgegevens toe aan de wachtrij
        self.queue.put(data)

    def _process(self):
        """
        Verwerk afbeeldingsgegevens en schrijf ze naar schijf.
        """
        while True:
            # Haal afbeeldingsgegevens uit de wachtrij
            filename, image_type, image = self.queue.get()

            # Schrijf de afbeelding naar schijf
            cv2.imwrite(
                os.path.join(
                    self.folder, image_type, f'{filename}.png'
                    ), image
                )

            # Markeer de taak als voltooid in de wachtrij
            self.queue.task_done()


class CanWorker:
    """
    Een werker die CAN-berichtwaarden naar schijf schrijft.
    """

    def __init__(self, can_queue: Queue, folder: str):
        """
        Initialisatie van de CanWorker.

        Parameters:
        - can_queue (Queue): Berichtenwachtrij voor CAN-berichten.
        - folder (str): Doelmap waar het CSV-bestand wordt opgeslagen.
        """
        # Object maken voor wachtrij
        self.queue = can_queue

        # Thread voor het verwerken van CAN-berichten
        self.thread = threading.Thread(
            target=self._process, args=(), daemon=True
            )

        # Doelmap voor het opslaan van het CSV-bestand
        self.folder_name = folder

        # Maak een nieuw CSV-bestand voor het opslaan van CAN-gegevens
        self.file_pointer = open(
            os.path.join(self.folder_name, 'recording.csv'), 'w'
            )

        # Schrijf de headerregel naar het CSV-bestand
        print(
            'Timestamp|'
            'Steering|'
            'SteeringSpeed|'
            'Throttle|'
            'Brake|'
            'SteeringSensor|'
            'SpeedSensor|'
            'Position',
            file=self.file_pointer
            )

    def start(self):
        """
        Start het verwerken van CAN-berichten.
        """
        # Start de verwerkingsthread
        self.thread.start()

    def stop(self):
        """
        Wacht tot alle CAN-berichten zijn verwerkt en sluit het CSV-bestand.
        """
        # Wacht tot alle taken in de wachtrij zijn voltooid
        self.queue.join()

        # Sluit het CSV-bestand
        self.file_pointer.close()

    def put(self, data):
        """
        Voegt CAN-berichtgegevens toe aan de wachtrij.

        Parameters:
        - data: CAN-berichtgegevens die moeten worden verwerkt.
        """
        # Voeg CAN-berichtgegevens toe aan de wachtrij
        self.queue.put(data)

    def _process(self):
        """
        Verwerk CAN-berichtgegevens en schrijf ze naar het CSV-bestand.
        """
        while True:
            # Haal CAN-berichtgegevens uit de wachtrij
            timestamp, values, position = self.queue.get()

            # Extract en formatteer stuurwaarde
            steering = str(
                struct.unpack(
                    "f", bytearray(values["steering"][:4])
                    )[0]
                    ) if values["steering"] else ""

            # Extract en formatteer stuursnelheidswaarde
            steering_speed = str(
                struct.unpack(
                    ">I", bytearray(values["steering"][4:])
                    )[0]
                    ) if values["steering"] else ""

            # Extract en formatteer gashendelwaarde
            throttle = str(
                values["throttle"][0]/100
                ) if values["throttle"] else ""

            # Extract en formatteer remwaarde
            brake = str(
                values["brake"][0]/100
                ) if values["brake"] else ""

            if values["steering_sensor"]:
                # Bereken stuurhoeksensorwaarde
                steering_sensor = (
                    values["steering_sensor"][1] << 8 |
                    values["steering_sensor"][2]
                    )

                # Verwerk de waarde om te zorgen dat deze correct is
                steering_sensor -= 65536 if steering_sensor > 32767 else 0
            else:
                # Als er geen stuurhoeksensorwaarden zijn, stel lege string in
                steering_sensor = ""

            # Extract en formatteer snelheidssensorwaarde
            speed_sensor = str(
                values["speed_sensor"][0]
                ) if values["speed_sensor"] else ""
            
            # Ophalen van postition
            pos = str(position)

            # Schrijf de geformatteerde gegevens naar het CSV-bestand
            print(
                f'{timestamp}|'
                f'{steering}|'
                f'{steering_speed}|'
                f'{throttle}|'
                f'{brake}|'
                f'{steering_sensor}|'
                f'{speed_sensor}|'
                f'{pos}',
                file=self.file_pointer
            )

            # Markeer de taak als voltooid in de wachtrij
            self.queue.task_done()


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


def initialize_can() -> can.Bus:
    """
    Initialiseert de CAN-bus interface en past filters
    toe voor de berichten waarin we geïnteresseerd zijn.

    Returns:
    - can.Bus: Geconfigureerde CAN-bus object.
    """
    # Initialiseer de CAN-bus
    bus = can.Bus(interface='socketcan', channel='can0', bitrate=500000)

    # Pas filters toe voor de CAN-berichten waarin we geïnteresseerd zijn
    bus.set_filters([
        {'can_id': 0x110, 'can_mask': 0xfff, 'extended': False},  # Rem
        {'can_id': 0x220, 'can_mask': 0xfff, 'extended': False},  # Stuur
        {'can_id': 0x330, 'can_mask': 0xfff, 'extended': False},  # Gas
        {'can_id': 0x1e5, 'can_mask': 0xfff, 'extended': False},  # Stuurhoek
        {'can_id': 0x440, 'can_mask': 0xfff, 'extended': False},  # Snelheid
    ])
    return bus


def main():
    # Geef een melding dat het initialiseren begint
    print('Initializing...', file=sys.stderr)

    # Initialiseer de CAN-bus
    bus = initialize_can()

    # Initialiseer de camera's
    cameras = initialize_cameras()

    # Geef een melding dat het maken van mappen begint
    print('Creating folders...', file=sys.stderr)

    # Maak een map met de huidige datum en tijd
    recording_folder = "recording " + datetime.now().strftime(
        "%d-%m-%Y %H-%M-%S"
        )

    # Controleer of de map al bestaat
    if not os.path.exists(recording_folder):
        # Maak de map aan als deze nog niet bestaat
        os.mkdir(recording_folder)
        # Maak submappen aan voor elk cameratype
        for subdir in cameras.keys():
            os.mkdir(os.path.join(recording_folder, subdir))

    # Maak een object voor het luisteren naar CAN-berichten
    can_listener = CanListener(bus)

    # Start het luisteren naar CAN-berichten
    can_listener.start_listening()

    # Maak een wachtrij voor afbeeldingsgegevens
    image_queue = Queue()

    # Maak een object voor het verwerken van afbeeldingen
    image_worker = ImageWorker(image_queue, recording_folder)

    # Start de verwerking van afbeeldingen
    ImageWorker(image_queue, recording_folder).start()
    ImageWorker(image_queue, recording_folder).start()

    # Start de hoofdverwerker voor afbeeldingen
    image_worker.start()

    # Maak een object voor het verwerken van CAN-berichten
    can_worker = CanWorker(Queue(), recording_folder)

    # Start het verwerken van CAN-berichten
    can_worker.start()

    # Initialiseer een dictionary voor framegegevens
    frames: Dict[str, cv2.Mat] = dict()

    # Aanzetten van de voorcamera voor rijden van de kart
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

        # Geef een melding dat het opnemen begint
        print('Recording...', file=sys.stderr)
        # Start running
        start_time = time.time()
        frame_count = 0  
        try:
            while True:
                _, frame = front_camera.read()

                steering_angle, throttle, brake, position = ll.process_image_to_instructie(frame)
    
                brake_msg.data = [int(99*max(0, brake))] + 7*[0]
                steering_msg.data = list(bytearray(struct.pack("f", float(steering_angle)))) + [0]*4
                throttle_msg.data = [int(99*max(0, throttle)), 0, 1] + 5*[0]
    
                brake_task.modify_data(brake_msg)
                steering_task.modify_data(steering_msg)
                throttle_task.modify_data(throttle_msg)

                # Houd het aantal succesvol opgehaalde frames bij
                ok_count = 0

                # Haal de nieuwste CAN-waarden op
                values = can_listener.get_new_values()

                # Haal de huidige timestamp op
                timestamp = time.time()

                for side, camera in cameras.items():
                    # Haal een frame op van de camera
                    ok, frames[side] = camera.retrieve()
                    ok_count += ok

                # Controleer of alle frames succesvol zijn opgehaald
                if ok_count == len(cameras):
                    for side, frame in frames.items():
                        # Voeg het frame toe aan de verwerkingswachtrij
                        image_worker.put((timestamp, side, frame))

                    # Voeg de CAN-waarden toe aan de verwerkingswachtrij
                    can_worker.put((timestamp, values, position))

                for camera in cameras.values():
                    # Ga door naar het volgende frame
                    camera.grab()

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

        # Stop het luisteren naar CAN-berichten
        can_listener.stop_listening()

        for camera in cameras.values():
            # Stop het vastleggen van videobeelden
            camera.release()

        # Stop de verwerking van afbeeldingen
        image_worker.stop()

        # Stop het verwerken van CAN-berichten
        can_worker.stop()

if __name__ == '__main__':
    main()
