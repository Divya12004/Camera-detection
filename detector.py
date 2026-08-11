import os
import csv
import time
import cv2
import pygame

from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO


# ==========================================
# LOAD .ENV
# ==========================================

load_dotenv()

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

CONFIDENCE_THRESHOLD = float(
    os.getenv("CONFIDENCE_THRESHOLD", "0.50")
)

ALARM_ENABLED = (
    os.getenv("ALARM_ENABLED", "true").lower() == "true"
)

ALARM_COOLDOWN = int(
    os.getenv("ALARM_COOLDOWN", "5")
)

SAVE_DETECTIONS = (
    os.getenv("SAVE_DETECTIONS", "true").lower() == "true"
)

DETECTION_DIR = os.getenv(
    "DETECTION_DIR",
    "detections"
)

LOG_DIR = os.getenv(
    "LOG_DIR",
    "logs"
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "yolov8n.pt"
)


# ==========================================
# DIRECTORIES
# ==========================================

os.makedirs(DETECTION_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ==========================================
# OBJECT CLASSES
# ==========================================

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


# ==========================================
# LOAD YOLO
# ==========================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# ==========================================
# ALARM
# ==========================================

pygame.mixer.init()

ALARM_FILE = "sounds/alarm.wav"

if ALARM_ENABLED:

    if os.path.exists(ALARM_FILE):

        pygame.mixer.music.load(ALARM_FILE)

        print("Alarm loaded successfully.")

    else:

        print("WARNING: alarm.wav not found.")

        ALARM_ENABLED = False


# ==========================================
# LOG FILE
# ==========================================

log_file = os.path.join(
    LOG_DIR,
    "detections.csv"
)

if not os.path.exists(log_file):

    with open(
        log_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "type",
            "animal",
            "confidence"
        ])


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(
    CAMERA_INDEX
)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


print("Camera started.")
print("Press Q to exit.")


# ==========================================
# IMPORTANT VARIABLES
# ==========================================

last_alarm = 0

last_detection = None

last_print_time = 0

PRINT_COOLDOWN = 3


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = camera.read()

    if not success:

        print("ERROR: Camera frame unavailable.")

        break


    detected_objects = []


    # ======================================
    # YOLO
    # ======================================

    results = model(
        frame,
        verbose=False
    )


    # ======================================
    # PROCESS RESULTS
    # ======================================

    for result in results:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            object_name = model.names[
                class_id
            ]

            confidence = float(
                box.conf[0]
            )


            # Confidence filter

            if confidence < CONFIDENCE_THRESHOLD:

                continue


            # Human / Animal filter

            if (
                object_name != HUMAN
                and object_name not in ANIMALS
            ):

                continue


            detected_objects.append(
                (
                    object_name,
                    confidence
                )
            )


            # Bounding box

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


            label = (
                f"{object_name} "
                f"{confidence * 100:.1f}%"
            )


            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    # ======================================
    # DETECTION STATE
    # ======================================

    current_detection = tuple(
        sorted(
            set(
                name
                for name, confidence
                in detected_objects
            )
        )
    )


    current_time = time.time()


    # ======================================
    # NEW DETECTION
    # ======================================

    if current_detection:

        if (
            current_detection != last_detection
            or
            current_time - last_print_time
            >= PRINT_COOLDOWN
        ):

            print()
            print("=" * 50)
            print("🚨 DETECTION ALERT")
            print("=" * 50)


            # Human + Animal

            if (
                HUMAN in current_detection
                and
                any(
                    x in ANIMALS
                    for x in current_detection
                )
            ):

                print(
                    "🚨 HUMAN + ANIMAL DETECTED!"
                )


            # Human

            elif HUMAN in current_detection:

                print(
                    "👤 HUMAN DETECTED!"
                )


            # Animal

            else:

                print(
                    "🐾 ANIMAL DETECTED!"
                )


            # Details

            for name, confidence in detected_objects:

                if name == HUMAN:

                    print(
                        f"   👤 Human"
                        f" | Confidence: "
                        f"{confidence * 100:.1f}%"
                    )

                else:

                    print(
                        f"   🐾 Animal: "
                        f"{name.upper()}"
                        f" | Confidence: "
                        f"{confidence * 100:.1f}%"
                    )


            print("=" * 50)


            last_detection = current_detection

            last_print_time = current_time


        # ==================================
        # ALARM
        # ==================================

        if (
            ALARM_ENABLED
            and
            current_time - last_alarm
            >= ALARM_COOLDOWN
        ):

            print("🔊 ALARM ACTIVATED!")

            pygame.mixer.music.play()

            last_alarm = current_time


    else:

        # Reset when nothing is detected

        if last_detection is not None:

            print()
            print("✅ No human or animal detected.")

        last_detection = None


    # ======================================
    # CAMERA TEXT
    # ======================================

    if detected_objects:

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

        cv2.putText(
            frame,
            "SCANNING...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "AI Human & Animal Detection",
        frame
    )


    # ======================================
    # EXIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

cv2.destroyAllWindows()

pygame.mixer.quit()

print("Application stopped.")
