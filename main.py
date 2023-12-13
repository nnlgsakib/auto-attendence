import cv2
import face_recognition
import os
import datetime
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMG_DIR = 'img/'
INFO_DIR = 'students_info/'
RECORD_DIR = 'record/'
COOLDOWN_PERIOD = 10
SKIP_FRAMES = 5

# Load student information and face encodings
def load_student_info(info_dir):
    student_info = {}
    for filename in os.listdir(info_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(info_dir, filename), 'r') as file:
                lines = file.readlines()
                name = lines[0].split(":")[1].strip()
                roll = lines[1].split(":")[1].strip()
                image_path = lines[2].split(":")[1].strip()
                student_image = face_recognition.load_image_file(image_path)
                student_encoding = face_recognition.face_encodings(student_image)[0]
                student_info[name] = {'roll': roll, 'encoding': student_encoding}
    return student_info

# Initialize camera
def initialize_camera():
    logger.info("Initializing camera...")
    return cv2.VideoCapture(0)

# Perform face recognition
def recognize_faces(frame, student_info, last_detection_time, recorded_students):
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    current_time = time.time()

    if current_time - last_detection_time >= COOLDOWN_PERIOD:
        for face_location, face_encoding in zip(face_locations, face_encodings):
            for name, info in student_info.items():
                match = face_recognition.compare_faces([info['encoding']], face_encoding)

                if match[0] and name not in recorded_students:
                    now = datetime.datetime.now()
                    timestamp = now.strftime('%Y-%m-%d')

                    record_filename = f"{timestamp}.txt"
                    record_filepath = os.path.join(RECORD_DIR, record_filename)

                    with open(record_filepath, 'a') as record_file:
                        record_file.write(f"name={name}\nroll={info['roll']}\npresent=true\ndate={now.strftime('%Y-%m-%d %H:%M:%S')}\n")

                    recorded_students.add(name)
                    last_detection_time = current_time
                    logger.info(f"Recorded student: {name}")

    return last_detection_time

# Record absent students
def record_absent_students(recorded_students, student_info):
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d')
    record_filename = f"{timestamp}.txt"
    record_filepath = os.path.join(RECORD_DIR, record_filename)

    with open(record_filepath, 'a') as record_file:
        for name, info in student_info.items():
            if name not in recorded_students:
                record_file.write(f"name={name}\nroll={info['roll']}\npresent=false\ndate={now.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Main loop
def main():
    student_info = load_student_info(INFO_DIR)
    cap = initialize_camera()
    last_detection_time = time.time()
    frame_count = 0
    recorded_students = set()

    logger.info("Starting main loop...")
    try:
        while True:
            ret, frame = cap.read()

            if frame_count % SKIP_FRAMES == 0:
                last_detection_time = recognize_faces(frame, student_info, last_detection_time, recorded_students)

            # Display a scanning animation
            cv2.putText(frame, "Scanning...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Display recorded students on the top left corner
            log_text = f"Recorded Students: {', '.join(recorded_students)}"

            cv2.putText(frame, log_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('Face Recognition', frame)

            # Use waitKey to control the display speed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                logger.info("Exiting main loop...")
                break

            frame_count += 1

    finally:
        cap.release()
        record_absent_students(recorded_students, student_info)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
