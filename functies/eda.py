# Importeren van libaries nodig voor functies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from IPython.display import display
import os
import cv2


def data_loader(folder_locatie):
    """
    Deze functie laad data in door middel van een folder locatie.
    Aan de hand van deze locatie kan de functie het CSV bestand en
    de afbeeldingen van elke camera inladen.

    Parameters:
    ----------
    folder_locatie : str
        Pad naar de folder op de SSD.

    Returns:
    ----------
    df : pd.DataFrame
        Een dataframe met de CSV bestanden.

    front : dict
        Een dictionary met de structuur bestandsnaam : afbeelding,
        voor de middelste camera.

    left : dict
        Een dictionary met de structuur bestandsnaam : afbeelding,
        voor de linker camera.

    right : dict
        Een dictionary met de structuur bestandsnaam : afbeelding,
        voor de rechter camera.
    """
    # Inladen van het csv bestand
    df = pd.read_csv(f"{folder_locatie}/recording.csv", sep="|")

    # Aanmaken dicts voor afbeeldingen
    front = {}
    left = {}
    right = {}

    # Inladen van de afbeeldingen
    for folder, image in [
            (f"{folder_locatie}/front", front),
            (f"{folder_locatie}/left", left),
            (f"{folder_locatie}/right", right)
            ]:
        print(f"Loading images from folder: {folder}")
        num_images_loaded = 0
        num_images_total = len(os.listdir(folder))
        for filename in os.listdir(folder):
            if filename.endswith('.png'):
                image_loc = os.path.join(folder, filename)
                try:
                    img = Image.open(image_loc)
                    image[filename] = img
                    num_images_loaded += 1
                    print(f"Afbeelding {num_images_loaded} geladen "
                          f"van {num_images_total} uit folder: {folder}")
                except Exception as e:
                    print(f"Error loading image {filename}: {e}")
        print(f"KLaar met laden van {num_images_loaded} afbeeldingen uit "
              f"{num_images_total} totale afbeeldingen uit folder: {folder}")

    return df, front, left, right


def image_overview(folder, image, sep="|"):
    """
    Deze functie toont dezelfde afbeelding vanuit elke folder, gepaard
    met de afbeelding grootte, modus en de bijbehorende data.

    Parameters:
    ----------
    folder : str
        Pad naar de folder met de afbeeldingen

    image : str of int
        De bestandsnaam, deze kan in zowel str als int gegeven worden

    sep : str (optional)
        De seperator van het csv bestand (default = "|")
    """
    try:
        # Kijken of de folder klopt
        if not os.path.exists(f"D:/{folder}"):
            raise FileNotFoundError("Folder niet gevonden.")

        # Kijken of alle bestanden aanwezig zijn
        if not os.path.exists(f"D:/{folder}/front/{image}.png"):
            raise FileNotFoundError(
                "Afbeelding niet gevonden in front directory."
                )

        if not os.path.exists(f"D:/{folder}/left/{image}.png"):
            raise FileNotFoundError(
                "Afbeelding niet gevonden in left directory."
                )

        if not os.path.exists(f"D:/{folder}/right/{image}.png"):
            raise FileNotFoundError(
                "Afbeelding niet gevonden in right directory."
                )

        if not os.path.exists(f"D:/{folder}/recording.csv"):
            raise FileNotFoundError(
                "CSV-bestand is niet gevonden."
                )

        # Als alles lukt, laad de afbeeldingen in
        img_front = Image.open(f"D:/{folder}/front/{image}.png")
        img_left = Image.open(f"D:/{folder}/left/{image}.png")
        img_right = Image.open(f"D:/{folder}/right/{image}.png")
        df = pd.read_csv(f"D:/{folder}/recording.csv", sep=sep)

    except FileNotFoundError as e:
        print(f"Bestand niet gevonden: {e}")

    except Exception as e:
        print(f"ERROR: {e}")

    # Ophalen van de dimensies van de afbeelding
    dimensions = {
        "Left": {"size": img_left.size, "mode": img_left.mode},
        "Front": {"size": img_front.size, "mode": img_front.mode},
        "Right": {"size": img_right.size, "mode": img_right.mode}
    }

    # Tonen van afbeeldingen en informatie
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (title, data) in zip(axs, dimensions.items()):
        ax.imshow(eval(f"img_{title.lower()}"))
        ax.axis('off')
        ax.set_title(f"{title}\nSize: {data['size']}\nMode: {data['mode']}")

    plt.tight_layout()
    plt.show()

    # Tonen van de bijbehorende data
    display(df[df["Timestamp"] == image])


def data_selection(df, start, end):
    """
    Selecteerd van een DataFrame de rijen tussen de aangegeven Timestamps.
    Zorg ervoor dat, bij gebruik van de functie, de Timestamp kolom in
    float format staat.

    Parameters:
    ----------
    df : pd.DataFrame
        Het pandas.DataFrame waar de Timestamp kolom aanwezig is

    start : int of float
        De int of float die de start-tijd aangeeft

    end : int of float
        De int of float die de eind-tijd aangeeft

    Returns:
    ----------
    data : pd.DataFrame
        Het dataframe waarbij alleen de data binnen de aangegeven tijden
        aanwezig is.
    """
    try:
        # Selecteer de data binnen de range van waarden
        data = df[(df['Timestamp'] >= start) & (df['Timestamp'] <= end)]
    except TypeError:
        raise TypeError("Fout bij data type. "
                        f"Start is {type(start)} en end is {type(end)}")

    return data


def lijngrafieken(df):
    """
    Deze functie toont een series van grafieken, gelijk aan het
    aantal kolommen dat geen "Timestamp" heet.

    Parameters:
    ----------
    df : pd.DataFrame
        Het dataframe dat de "Timestamp" kolom bevat
    """
    # Bepaal de grootte van de grafiek
    plt.figure(figsize=(15, 9))

    # Berekenen aan kolommen - "Timestamp"
    num_cols = len(df.columns) - 1

    # Loop over de kolommen, behalve "Timestamp"
    for i, column in enumerate(df.columns):
        if column != "Timestamp":
            # Genereer een subplot op basis van het aantal kolommen
            ax = plt.subplot(num_cols, 1, i)

            # Maken van een lineplot
            sns.lineplot(data=df, x="Timestamp", y=column, ax=ax)

            # Labelen van de assen
            plt.xlabel("Timestamp")
            plt.ylabel(column, rotation=0)
            ax.yaxis.set_label_coords(-0.12, 0.5)

    # Aanpassen van de layout voor mooiere grafieken
    plt.suptitle("Grafieken van opgevraagde ronde", fontsize=16, y=1)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.9)
    plt.show()


def scaled_vergelijking(df, kolom1="Steering", kolom2="SteeringSensor"):
    """
    Deze functie scaled de data tussen -1 en 1 om zo de kolommen
    gemakkelijk te kunnen vergelijken. De standaard kolommen zijn
    "Steering" en "SteeringSensor" omdat deze kolommen normaal erg
    verschillende waarden hebben. Op deze manier konden ze gemakkelijk
    vergeleken worden.

    Parameters:
    ----------
    df : pd.DataFrame
        Het dataframe dat de "Timestamp" kolom en de te vergelijken
        kolommen bevat

    kolom1 : str (optioneel) default="Steering"
        De kolomnaam van kolom 1

    kolom2 : str (optioneel) default="SteeringSensor"
        De kolomnaam van kolom 2
    """
    # Copy het originele dataframe
    scaled = df.copy()

    # Scale de data tussen -1 en 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled[[kolom1, kolom2]] = scaler.fit_transform(scaled[[kolom1, kolom2]])

    # Bepaal de grootte van de grafiek
    plt.figure(figsize=(20, 6))

    # Plot de lijnen van de grafiek
    sns.lineplot(
        data=scaled, x="Timestamp", y=kolom1, color='orange', label=kolom1
        )
    sns.lineplot(
        data=scaled, x="Timestamp", y=kolom2, color='green', label=kolom2
        )

    # Labelen van de assen en titel
    plt.xlabel("Timestamp")
    plt.ylabel("Waarde van kolommen (scaled)")
    plt.title(f"Vergelijking tussen {kolom1} en {kolom2}")

    # Aanmaken legenda onder titel
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend_.remove()
    plt.figlegend(
        handles, labels, loc='upper center',
        ncol=2, bbox_to_anchor=(0.515, 1), frameon=False
        )
    plt.subplots_adjust(top=0.9)
    plt.show()


def load_images(df, image_folder):
    """
    Deze functie laad de afbeelding in die overeenkomen met de Timestamps
    van het gekozen dataframe. Als je de functie tweemaal runt op hetzelfde
    gedeelte van een dataframe, zal de functie niet werken omdat de
    afbeelding al is ingeladen.

    Parameters:
    ----------
    df : pd.DataFrame
        Het pandas.DataFrame waarvan je de foto's wilt ophalen

    image_folder : str
        Het pad naar de afbeelding map die je wilt inladen.

    Returns:
    ----------
    images : dict
        Een dictionary waarbij de keys de afbeelding namen zijn
        en de values de afbeeldingen.
    """
    # Maken van image dict
    images = {}

    # Loopen over df voor timestamps
    for index, row in df.iterrows():
        timestamp = str(row['Timestamp'])
        # Maken van de filepath
        image_path = f"{image_folder}/{timestamp}.png"
        try:
            # Inladen van de image en opslaan
            img = Image.open(image_path)
            images[timestamp] = img
        except FileNotFoundError:
            pass

    print(f"{len(images)} / {len(df)} afbeeldingen zijn ingeladen.")
    return images


def img_to_vid(images, output_vid_pad, df, fps=30.0):
    """
    Deze functie is in staat om afbeelding om te zetten in een video.
    Deze video kan eventueel voor andere doeleinden worden gebruikt.

    Parameters:
    ----------
    images : dict
        Een dictionary met afbeeldingen. Deze afbeeldingen zijn
        volledig ingeladen opgeslagen.

    output_vid_pad : str
        Pad en naam van de uiteindelijke video. Let op de .mp4
        extensie aan het einde van de naam!!

    data : pandas DataFrame
        DataFrame containing the data to be overlaid on the video frames.
        It should have columns: 'Timestamp', 'Throttle',
        'Brake', 'Steering', 'SteeringSensor'.

    fps : float (optional)
        Een float die het aantal frames per seconde weergeeft
        (default = 30.0)
    """
    # Filtreren data op images
    timestamps = [float(timestamp) for timestamp in images.keys()]
    data = df[df['Timestamp'].isin(timestamps)]

    # Verkrijg de dimensies van de afbeeldingen
    timestamp, first_image = next(iter(images.items()))
    width, height = first_image.size
    size = (width, height)

    # Aanmaken van de videowriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_vid_pad, fourcc, fps, size)

    # Itereren door de afbeeldingen en schrijf ze naar de video
    for timestamp, image in images.items():
        # omzetten van afbeelding naar numpy array om frame te maken
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Zetten van text over het filmpje
        timestamp_data = data[data['Timestamp'] == float(timestamp)].iloc[0]
        text = f"Throttle: {timestamp_data['Throttle']} " \
               f"Brake: {timestamp_data['Brake']} " \
               f"Steering: {timestamp_data['Steering']}"
        cv2.putText(
            frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2, cv2.LINE_AA
            )

        out.write(frame)

    # Finish!!!!
    out.release()
