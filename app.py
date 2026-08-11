import os
import cv2
import time
import csv

from datetime import datetime

from flask import Flask, Response, render_template, jsonify
from dotenv import load_dotenv
from ultralytics import YOLO


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

HOST = os.getenv(
    "HOST",
    "0.0.0.0"
)

PORT = int(
    os.getenv(
        "PORT",
        "5000"
    )
)

CAMERA_INDEX = int(
    os.getenv(
        "CAMERA_INDEX",
        "0"
    )
)

CONFIDENCE_THRESHOLD = float(
    os.getenv(
        "CONFIDENCE_THRESHOLD",
        "0.50"
    )
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "yolov8n.pt"
)

SAVE_DETECTIONS = (
    os.getenv(
        "SAVE_DETECTIONS",
        "true"
    ).lower() == "true"
)

DETECTION_DIR = os.getenv(
    "DETECTION_DIR",
    "detections"
)

LOG_DIR = os.getenv(
    "LOG_DIR",
    "logs"
)


# ============================================================
# IMAGE RETENTION
# ============================================================

IMAGE_RETENTION_DAYS = 5


# ============================================================
# DETECTION EVENT SETTINGS
# ============================================================

EVENT_RESET_SECONDS = 5


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    DETECTION_DIR,
    exist_ok=True
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)


# ============================================================
# LOG FILE
# ============================================================

LOG_FILE = os.path.join(
    LOG_DIR,
    "detections.csv"
)


# ============================================================
# CREATE CSV HEADER
# ============================================================

if not os.path.exists(LOG_FILE):

    with open(
        LOG_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "type",
            "object",
            "confidence",
            "image"
        ])


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# YOLO OBJECT CLASSES
# ============================================================

HUMAN = "person"


ANIMALS = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe"
}


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print()
print("=" * 60)
print("Loading YOLO model...")
print("=" * 60)

model = YOLO(
    MODEL_PATH
)

print("✅ YOLO model loaded successfully.")


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(
    CAMERA_INDEX
)


if not camera.isOpened():

    print(
        "❌ ERROR: Could not open camera."
    )

    raise RuntimeError(
        "Camera could not be opened."
    )


print("✅ Camera opened successfully.")


# ============================================================
# CURRENT DETECTION STATUS
# ============================================================

current_status = {

    "detected": False,

    "type": "NONE",

    "object": "",

    "confidence": 0,

    "image": "",

    "timestamp": ""
}


# ============================================================
# DETECTION EVENT TRACKING
# ============================================================

last_detection_event = None

last_saved_time = 0


# ============================================================
# DELETE OLD IMAGES
# ============================================================

def cleanup_old_images():

    cutoff_time = (
        time.time()
        -
        (
            IMAGE_RETENTION_DAYS
            *
            24
            *
            60
            *
            60
        )
    )


    deleted_count = 0


    try:

        files = os.listdir(
            DETECTION_DIR
        )

    except OSError as error:

        print(
            f"❌ Could not read detection "
            f"directory: {error}"
        )

        return


    for filename in files:

        # Only image files

        if not filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        ):

            continue


        file_path = os.path.join(
            DETECTION_DIR,
            filename
        )


        try:

            file_time = os.path.getmtime(
                file_path
            )


            if file_time < cutoff_time:

                os.remove(
                    file_path
                )

                deleted_count += 1


        except OSError as error:

            print(
                f"⚠️ Could not delete "
                f"{file_path}: {error}"
            )


    if deleted_count > 0:

        print(
            f"🗑️ Deleted "
            f"{deleted_count} "
            f"old detection image(s)."
        )


# ============================================================
# SAVE DETECTION IMAGE
# ============================================================

def save_detection_image(
    frame,
    detection_type,
    detected_objects
):

    timestamp = datetime.now()


    timestamp_string = (
        timestamp.strftime(
            "%Y%m%d_%H%M%S"
        )
    )


    image_filename = (
        f"detection_"
        f"{timestamp_string}.jpg"
    )


    image_path = os.path.join(
        DETECTION_DIR,
        image_filename
    )


    # Save image

    success = cv2.imwrite(
        image_path,
        frame
    )


    if success:

        print(
            f"📸 Detection image saved: "
            f"{image_path}"
        )

    else:

        print(
            "❌ Failed to save detection image."
        )

        image_filename = ""


    # ========================================================
    # SAVE CSV LOG
    # ========================================================

    try:

        with open(
            LOG_FILE,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(
                file
            )


            for (
                object_name,
                confidence
            ) in detected_objects:


                if object_name == HUMAN:

                    object_type = "HUMAN"


                else:

                    object_type = "ANIMAL"


                writer.writerow([

                    timestamp.isoformat(),

                    object_type,

                    object_name,

                    round(
                        confidence * 100,
                        1
                    ),

                    image_filename

                ])


    except OSError as error:

        print(
            f"❌ Could not write log: "
            f"{error}"
        )


    return image_filename


# ============================================================
# GENERATE CAMERA FRAMES
# ============================================================

def generate_frames():

    global current_status

    global last_detection_event

    global last_saved_time


    while True:

        # ====================================================
        # READ CAMERA
        # ====================================================

        success, frame = camera.read()


        if not success:

            print(
                "❌ Could not read camera frame."
            )

            break


        # ====================================================
        # DETECT OBJECTS
        # ====================================================

        detected_objects = []


        results = model(
            frame,
            verbose=False
        )


        # ====================================================
        # PROCESS YOLO RESULTS
        # ====================================================

        for result in results:


            for box in result.boxes:


                class_id = int(
                    box.cls[0]
                )


                object_name = (
                    model.names[
                        class_id
                    ]
                )


                confidence = float(
                    box.conf[0]
                )


                # --------------------------------------------
                # CONFIDENCE FILTER
                # --------------------------------------------

                if (
                    confidence
                    <
                    CONFIDENCE_THRESHOLD
                ):

                    continue


                # --------------------------------------------
                # HUMAN / ANIMAL FILTER
                # --------------------------------------------

                if (

                    object_name != HUMAN

                    and

                    object_name not in ANIMALS

                ):

                    continue


                # --------------------------------------------
                # SAVE DETECTION
                # --------------------------------------------

                detected_objects.append(

                    (
                        object_name,
                        confidence
                    )

                )


                # --------------------------------------------
                # BOUNDING BOX
                # --------------------------------------------

                x1, y1, x2, y2 = map(

                    int,

                    box.xyxy[0]

                )


                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    2

                )


                # --------------------------------------------
                # LABEL
                # --------------------------------------------

                label = (

                    f"{object_name.upper()} "

                    f"{confidence * 100:.1f}%"

                )


                cv2.putText(

                    frame,

                    label,

                    (
                        x1,
                        max(
                            y1 - 10,
                            20
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0, 255, 0),

                    2

                )


        # ====================================================
        # PROCESS DETECTION STATUS
        # ====================================================

        if detected_objects:


            # ----------------------------------------------
            # UNIQUE DETECTED OBJECTS
            # ----------------------------------------------

            detected_names = tuple(

                sorted(

                    set(

                        name

                        for (
                            name,
                            confidence
                        )

                        in detected_objects

                    )

                )

            )


            # ----------------------------------------------
            # HUMAN DETECTED
            # ----------------------------------------------

            human_detected = (

                HUMAN

                in

                detected_names

            )


            # ----------------------------------------------
            # ANIMAL DETECTED
            # ----------------------------------------------

            animal_detected = any(

                name in ANIMALS

                for name

                in detected_names

            )


            # ----------------------------------------------
            # DETECTION TYPE
            # ----------------------------------------------

            if (

                human_detected

                and

                animal_detected

            ):

                detection_type = (
                    "HUMAN + ANIMAL"
                )


            elif human_detected:

                detection_type = (
                    "HUMAN"
                )


            else:

                detection_type = (
                    "ANIMAL"
                )


            # ----------------------------------------------
            # BEST CONFIDENCE
            # ----------------------------------------------

            best_name, best_confidence = max(

                detected_objects,

                key=lambda item: item[1]

            )


            # ----------------------------------------------
            # UPDATE STATUS
            # ----------------------------------------------

            current_status = {

                "detected": True,

                "type": detection_type,

                "object": best_name,

                "confidence": round(

                    best_confidence * 100,

                    1

                ),

                "image": "",

                "timestamp":
                    datetime.now().isoformat()

            }


            # =================================================
            # DETECTION EVENT
            # =================================================

            current_time = time.time()


            new_detection = (

                detected_names

                !=

                last_detection_event

            )


            enough_time_passed = (

                current_time

                -

                last_saved_time

                >=

                EVENT_RESET_SECONDS

            )


            # =================================================
            # SAVE ONLY ONE IMAGE PER EVENT
            # =================================================

            if (

                SAVE_DETECTIONS

                and

                (

                    new_detection

                    or

                    enough_time_passed

                )

            ):


                image_filename = (
                    save_detection_image(

                        frame,

                        detection_type,

                        detected_objects

                    )

                )


                current_status[
                    "image"
                ] = image_filename


                last_detection_event = (
                    detected_names
                )


                last_saved_time = (
                    current_time
                )


                # ---------------------------------------------
                # DELETE OLD IMAGES
                # ---------------------------------------------

                cleanup_old_images()


            # =================================================
            # CAMERA ALERT TEXT
            # =================================================

            cv2.putText(

                frame,

                "ALERT: DETECTED",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 0, 255),

                2

            )


        else:


            # =================================================
            # NO DETECTION
            # =================================================

            current_status = {

                "detected": False,

                "type": "NONE",

                "object": "",

                "confidence": 0,

                "image": "",

                "timestamp":
                    datetime.now().isoformat()

            }


            # Reset event

            last_detection_event = None


            # Camera status

            cv2.putText(

                frame,

                "SCANNING...",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (255, 255, 255),

                2

            )


        # ====================================================
        # ENCODE FRAME
        # ====================================================

        success, buffer = cv2.imencode(

            ".jpg",

            frame

        )


        if not success:

            continue


        frame_bytes = (
            buffer.tobytes()
        )


        # ====================================================
        # SEND FRAME TO BROWSER
        # ====================================================

        yield (

            b"--frame\r\n"

            b"Content-Type: "
            b"image/jpeg\r\n\r\n"

            + frame_bytes

            + b"\r\n"

        )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# LIVE VIDEO
# ============================================================

@app.route("/video")
def video():

    return Response(

        generate_frames(),

        mimetype=(

            "multipart/x-mixed-replace; "

            "boundary=frame"

        )

    )


# ============================================================
# DETECTION STATUS API
# ============================================================

@app.route("/status")
def status():

    return jsonify(
        current_status
    )


# ============================================================
# DETECTION HISTORY API
# ============================================================

@app.route("/history")
def history():

    records = []


    if os.path.exists(
        LOG_FILE
    ):

        try:

            with open(

                LOG_FILE,

                "r",

                newline=""

            ) as file:


                reader = csv.DictReader(
                    file
                )


                for row in reader:

                    records.append(
                        row
                    )


        except OSError as error:

            print(
                f"❌ Could not read "
                f"log file: {error}"
            )


    # Latest 20 records

    records = records[-20:]


    # Newest first

    records.reverse()


    return jsonify(
        records
    )


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":


    print()

    print("=" * 60)

    print(
        "🤖 AI HUMAN & ANIMAL DETECTION"
    )

    print("=" * 60)

    print(
        f"📷 Camera Index: "
        f"{CAMERA_INDEX}"
    )

    print(
        f"🎯 Confidence: "
        f"{CONFIDENCE_THRESHOLD}"
    )

    print(
        f"📸 Save Images: "
        f"{SAVE_DETECTIONS}"
    )

    print(
        f"🗂️ Image Folder: "
        f"{DETECTION_DIR}"
    )

    print(
        f"🗑️ Retention: "
        f"{IMAGE_RETENTION_DAYS} days"
    )

    print(
        f"🌐 Port: "
        f"{PORT}"
    )

    print("=" * 60)

    print()

    print(
        f"🌐 Open browser:"
    )

    print(
        f"http://localhost:{PORT}"
    )

    print()


    app.run(

        host=HOST,

        port=PORT,

        debug=False,

        threaded=True

    )
