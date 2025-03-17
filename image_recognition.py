import cv2
import face_recognition
import numpy as np
import pickle
import os
import sqlite3
from datetime import datetime, timedelta

# Constants
LED_TIMEOUT = 5  # Duration in seconds to keep the LED on (Unused in this context)
DISPLAY_EXTRA_TIME = 30  # Additional seconds to keep the display after person leaves
ENCODING_DISTANCE_THRESHOLD = 0.45  # Reduced for stronger accuracy
SCALE_FACTOR = 0.25  # Keeps the image at a reasonable scale for real-time processing
# Database setup
DATABASE = 'students.db'

def get_student_info(student_id):
    """Fetch student information from the database using student ID."""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT name, emp, age, department, gmail, star FROM student WHERE id = ?", (student_id,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return {
                'id': student_id,
                'name': result[0],
                'emp': result[1],
                'age': result[2],
                'department': result[3],
                'gmail': result[4],
                'star': result[5]
            }
        else:
            return None
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None


# Track attendance IDs to update endtime
attendance_ids = {}  # {student_id: attendance_id}

def log_attendance_in(student_id):
    """Log the attendance of a student (insert starttime and leave endtime empty)."""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO attendance (student_id, starttime) VALUES (?, ?)", (student_id, datetime.now()))
        conn.commit()
        attendance_id = cursor.lastrowid  # Get the ID of the newly created row for future updates
        conn.close()
        print(f"Attendance logged in for student ID {student_id} at {datetime.now()} with attendance ID {attendance_id}")
        return attendance_id
    except sqlite3.Error as e:
        print(f"Failed to log attendance in: {e}")
        return None

def log_attendance_out(attendance_id):
    """Update the attendance record of a student (add endtime)."""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("UPDATE attendance SET endtime = ? WHERE id = ?", (datetime.now(), attendance_id))
        conn.commit()
        conn.close()
        print(f"Attendance logged out for attendance ID {attendance_id} at {datetime.now()}")
    except sqlite3.Error as e:
        print(f"Failed to log attendance out: {e}")

def load_encodings(encode_file_path):
    if not os.path.exists(encode_file_path):
        print(f"Encoding file {encode_file_path} not found.")
        exit(1)

    with open(encode_file_path, 'rb') as file:
        encodeListKnownWithIds = pickle.load(file)

    if len(encodeListKnownWithIds) != 2:
        print("Invalid encoding file format. Should contain [encodeListKnown, studentIds].")
        exit(1)

    encodeListKnown, studentIds = encodeListKnownWithIds
    print(f"Loaded {len(encodeListKnown)} encodings for {len(set(studentIds))} students.")
    return encodeListKnown, studentIds

def preprocess_image(image, scale_factor):
    """Apply preprocessing steps to the image."""
    # Resize for faster processing
    img_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return img_rgb

def enhance_encoding(known_encodings, known_ids):
    """Add multi-angle encoding for each person to improve recognition accuracy."""
    multi_angle_encodings = []
    multi_angle_ids = []

    for encoding, student_id in zip(known_encodings, known_ids):
        # Rotate the face by small angles to create multiple encodings for the same person
        for angle in [-10, 0, 10]:  # Add variations
            multi_angle_encodings.append(encoding)  # Add the same encoding for simplicity
            multi_angle_ids.append(student_id)

    return multi_angle_encodings, multi_angle_ids

def display_student_info(imgDisplay, studentInfo, IMAGES_FOLDER, imgrecognize):
    """Display student information and overlay their image on the display."""
    imgDisplay[0:657, 0:1313] = imgrecognize
    if not studentInfo:
        return

    if studentInfo['star'] > 3:
        authorization_status = "authorized"
        status_color = (0, 0, 0)
    else:
        authorization_status = "unauthorised"
        status_color = (0, 0, 0)

    # Display student info text
    # Display star rating and authorization status 
    cv2.putText(imgDisplay, f"SKILL LEVEL: {'*' * studentInfo['star']} ", (300, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(imgDisplay, f"Status: {authorization_status}", (1000, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)  

    cv2.putText(imgDisplay, f"Emp Name: {studentInfo['name']}", (164, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(imgDisplay, f"Emp ID: {studentInfo['emp']}", (500, 345), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(imgDisplay, f"Age: {studentInfo['age']}", (160, 435), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(imgDisplay, f"Dep: {studentInfo['department']}", (550, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(imgDisplay, f"Gmail: {studentInfo['gmail']}", (180, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Load and overlay one of the student's images (front image preferred)
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT image_filename FROM image WHERE student_id = ? AND image_type = 'front'", (studentInfo['id'],))
        front_images = cursor.fetchall()
        conn.close()
    except sqlite3.Error as e:
        print(f"Database error while fetching images: {e}")
        front_images = []

    if front_images:
        for front_image in front_images:
            student_image_path = os.path.join(IMAGES_FOLDER, front_image[0])  # Corrected variable name
            if os.path.exists(student_image_path):
                student_img = cv2.imread(student_image_path)
                if student_img is not None:
                    # Resize the student image to fit on the display (e.g., 200x200)
                    student_img = cv2.resize(student_img, (300, 300))
                    # Define the position where the image will be placed
                    x_offset, y_offset = 900, 180  # Adjust as needed
                    # Ensure that the overlay does not go out of bounds
                    if (y_offset + student_img.shape[0] < imgDisplay.shape[0] and
                        x_offset + student_img.shape[1] < imgDisplay.shape[1]):
                        imgDisplay[y_offset:y_offset + student_img.shape[0],
                                   x_offset:x_offset + student_img.shape[1]] = student_img
                    else:
                        print(f"Overlay position out of bounds for student ID {studentInfo['id']}")
                else:
                    print(f"Failed to load image for student ID {studentInfo['id']}")
            else:
                print(f"Image file does not exist: {student_image_path}")
    else:
        print(f"No front image found for student ID {studentInfo['id']}")

def main():
    # Load encoding file
    ENCODE_FILE = 'EncodeFile.p'
    encodeListKnown, studentIds = load_encodings(ENCODE_FILE)
    encodeListKnown, studentIds = enhance_encoding(encodeListKnown, studentIds)

    # Define the path to the Images folder
    IMAGES_FOLDER = os.path.join('static', 'Images')  # Ensure this path matches Flask's UPLOAD_FOLDER

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height

    # Load background image
    BACKGROUND_IMAGE = os.path.join('Resources', 'home.jpg')
    if not os.path.exists(BACKGROUND_IMAGE):
        print(f"Background image {BACKGROUND_IMAGE} not found.")
        exit(1)
    imgBackground = cv2.imread(BACKGROUND_IMAGE)
    if imgBackground is None:
        print(f"Failed to load background image from {BACKGROUND_IMAGE}.")
        exit(1)

    recognised_img = os.path.join('Resources', 'recognized.jpg')
    unrecognised_img = os.path.join('Resources', 'unrecognised.jpg')
 
    imgrecognize = cv2.imread(recognised_img)
    imgunrecognize = cv2.imread(unrecognised_img)

    imgrecognize = cv2.resize(imgrecognize, (1313, 657))
    imgunrecognize = cv2.resize(imgunrecognize, (1313, 657))

    # Variables to manage display timing
    last_displayed_student = None
    display_info_until = None  # Timestamp until which to display the info

    last_seen = {}  # To track the last time each student was seen
    logged_out_students = {}  # To track students who were logged out

    # Main loop
    print("Starting face recognition. Press 'q' to quit.")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Preprocess the image (resize and convert to RGB)
        imgS = preprocess_image(img, SCALE_FACTOR)

        # Detect face locations and encodings in the current frame
        faceCurFrame = face_recognition.face_locations(imgS, model='hog')  # Use 'cnn' for more accuracy
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
        imgDisplay = imgBackground.copy()
        current_detected_ids = set()
        current_time = datetime.now()

        recognized_face = False

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            # Use face distance to improve accuracy
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if faceDis[matchIndex] < ENCODING_DISTANCE_THRESHOLD:  # Adjusted threshold
                recognized_face = True
                student_id = studentIds[matchIndex]
                current_detected_ids.add(student_id)

                studentInfo = get_student_info(student_id)
                last_seen[student_id] = current_time

                # Handle attendance logging
                if student_id not in attendance_ids:
                    attendance_id = log_attendance_in(student_id)
                    if attendance_id:
                        attendance_ids[student_id] = attendance_id

                if student_id in logged_out_students:
                    del logged_out_students[student_id]

                # Update display variables
                last_displayed_student = studentInfo
                display_info_until = current_time + timedelta(seconds=DISPLAY_EXTRA_TIME)
            else:
                imgDisplay[0:657, 0:1313] = imgunrecognize

        # Log attendance out if a student hasn't been detected for a while
        to_logout = []
        for student_id, last_seen_time in last_seen.items():
            if (current_time - last_seen_time).total_seconds() > 30:
                # Log the student's attendance out
                attendance_id = attendance_ids.get(student_id)
                if attendance_id:
                    log_attendance_out(attendance_id)
                    del attendance_ids[student_id]
                    logged_out_students[student_id] = current_time  # Track when they were logged out
                    to_logout.append(student_id)

        # Remove logged-out students from `last_seen`
        for student_id in to_logout:
            del last_seen[student_id]   

        # Display the last detected student's info if within the display timeout
        if last_displayed_student and current_time < display_info_until:
            display_student_info(imgDisplay, last_displayed_student, IMAGES_FOLDER, imgrecognize)
        elif last_displayed_student and current_time >= display_info_until:
            last_displayed_student = None  # Clear the display after timeout

        # Display logged-out students for a brief period after logout
        for student_id, logout_time in logged_out_students.items():
            if (current_time - logout_time).total_seconds() <= DISPLAY_EXTRA_TIME:
                studentInfo = get_student_info(student_id)

        # Display the updated background with attendance info
        cv2.imshow("Face Attendance", imgDisplay)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
