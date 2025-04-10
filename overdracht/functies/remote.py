import easyocr
import cv2
from ultralytics import YOLO
from IPython.display import Image
import os
from PIL import Image

print("Loading YOLO Model...")
reader = easyocr.Reader(['en'])
model = YOLO("../obj_model.pt")

def process_image(input, max_throttle, speed_hmu):
    img = input
    img_obb = img
    results = model.predict(img)
    img = Image.fromarray(img_obb) 

    # Draw bounding boxes on the image
    for result in results:
        boxes = result.boxes.xyxy  # Extract bounding box coordinates
        classes = result.boxes.cls.cpu().numpy()  # Extract class IDs
        confidences = result.boxes.conf.cpu().numpy()  # Extract confidences
        class_to_label = result.names
        
        # Iterate over each detected object
        for box, class_id, confidence in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            if class_id in class_to_label:
                label_name = class_to_label[class_id] + " " + str(round(confidence, 2))  # Get label name from dictionary
            else:
                label_name = "Unknown" + " " + str(round(confidence, 2))  # If class ID not found in dictionary
            # Draw bounding box and label on the image
            cv2.rectangle(img_obb, (x1, y1), (x2, y2), (0, 122, 255), 2)
            cv2.putText(img_obb, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 122, 255), 2)

    for box in results[0].boxes:
        obj = box.cls.tolist()[0]

        if obj == 1: # rood
            x1, y1, x2, y2 = box.xyxy[0]
            width = (x2 - x1).item()
            length = (y2 - y1).item()
            
            if length > 30: #rood stoplicht staat dichtbij, dus kart moet remmen
                throttle = 0
                brake = 1
                #cv2.putText(img_obb, 'rood! sta stil', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            elif length > 20: #rood stoplicht is er bijna, stopt met gas geven
                throttle = 0
                brake = 0
                #cv2.putText(img_obb, 'pas op rood', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        elif obj == 4: # snelheid
            x1, y1, x2, y2 = box.xyxy[0]
            width = (x2 - x1).item()
            length = (y2 - y1).item()
    
            if width > 40: # als snelheidsboord op een afstand staat dat je het kan aflezen. lees het af
                sign_array = np.array(img.crop((int(x1), int(y1), int(x2), int(y2))))
        
                # Lezen van de np.array met EasyOCR
                speed_limit = reader.readtext(sign_array)

                try:
                    # Ophalen van de snelheid en omzetten naar hm/u
                    speed_hmu = int(speed_limit[0][1]) * 10
                    max_throttle = speed_to_throttle(speed_hmu)
                except (ValueError, IndexError, TypeError):
                    pass 

        elif obj == 0: # zebra
            x1, y1, x2, y2 = box.xyxy[0]
            width = (x2 - x1).item()
            length = (y2 - y1).item()
            #cv2.putText(img_obb, f"zebra: {width, length}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            if length > 100: #staat recht voor zebrapad, dus kart moet remmen
                throttle = 0
                brake = 1
                cv2.putText(img_obb, 'Zebrapad!', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            elif length > 30: #zebrapad is er bijna, stopt met gas geven
                throttle = 0
                brake = 0
                cv2.putText(img_obb, 'pas op Zebrapad', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        if obj == 3: # MENSEN
            x1, y1, x2, y2 = box.xyxy[0]
            width = (x2 - x1).item()
            length = (y2 - y1).item()
            #cv2.putText(img_obb, f"mensen: {width, length}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            #cv2.putText(img_obb, f"x1, y1, x2, y2: {x1, y1, x2, y2}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        if obj == 5: # auto
            x1, y1, x2, y2 = box.xyxy[0]
            width = (x2 - x1).item()
            length = (y2 - y1).item()
            #cv2.putText(img_obb, f"auto: {width, length}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            #cv2.putText(img_obb, f"x1, y1, x2, y2: {x1, y1, x2, y2}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.putText(img_obb, f"max throttle: {max_throttle} max speed: {speed_hmu}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_obb, max_throttle, speed_hmu


def process_folder(folder, base_folder_path, output_folder, speed_hmu, max_throttle):
    """
    Verwerkt alle afbeeldingen in een opgegeven map door elke afbeelding te lezen, 
    te verwerken en de verwerkte afbeeldingen op te slaan in een uitvoermap.

    Parameters:
    folder (str): De naam van de submap met opnames.
    base_folder_path (str): Het basispad waar de submap zich bevindt.
    output_folder (str): Het pad naar de map waar de verwerkte afbeeldingen worden opgeslagen.
    speed_hmu (int): De initiële snelheid voor de verwerking van de afbeelding.
    max_throttle (int): De maximale gaspositie voor de verwerking van de afbeelding.

    Returns:
    None
    """
    folder1_path = os.path.join(base_folder_path, folder, 'front')
    folder1_images = sorted(os.listdir(folder1_path))

    for imgaa in folder1_images:
        file_path = os.path.join(folder1_path, imgaa)
        file_path = os.path.normpath(file_path)
        
        img_test = cv2.imread(file_path)
        
        fototje, max_throttle, speed_hmu = process_image(img_test, max_throttle, speed_hmu)
        cv2.imwrite(os.path.join(output_folder, f"test_{imgaa}"), fototje)


def create_video_from_images(image_folder, video_name, frame_rate=30):
    """
    Maakt een video van afbeeldingen in een opgegeven map.

    Parameters:
    image_folder (str): Het pad naar de map die de afbeeldingen bevat.
    video_name (str): De naam van het uitvoervideobestand.
    frame_rate (int): Het aantal frames per seconde voor de video (standaard is 30).

    Returns:
    None
    """
    # Filter afbeeldingen in de map en sorteer ze op naam
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    
    if not images:
        raise ValueError("Geen .png-afbeeldingen gevonden in de opgegeven map.")

    # Lees de eerste afbeelding om de afmetingen te verkrijgen
    first_frame_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_frame_path)
    if frame is None:
        raise ValueError(f"Kan de eerste afbeelding niet lezen: {first_frame_path}")

    height, width, layers = frame.shape

    # Maak een VideoWriter-object
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    # Voeg elke afbeelding toe aan de video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Kan afbeelding niet lezen: {image_path}")
        video.write(frame)

    # Release de VideoWriter
    video.release()