import os 
import cv2
import face_recognition
import os
import glob
import numpy as np
import datetime
import sqlite3
# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Connect to SQLite database (create it if it doesn't exist)
conn = sqlite3.connect("database.db")
cursor = conn.cursor()

# Create the RecognitionLog table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS RecognitionLog (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Name TEXT NOT NULL,
        Timestamp TEXT NOT NULL
    )
''')
conn.commit()

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*"))
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing

        # Draw rectangles and blur faces for recognized individuals
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top = int(top)
            right = int(right)
            bottom = int(bottom)
            left = int(left)

            if name != "Unknown":
                # Draw rectangle
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Blur face region
                face_roi = frame[top:bottom, left:right]
                face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)

                # Place blurred face back into the frame
                frame[top:bottom, left:right] = face_roi

        return face_locations.astype(int), face_names


# Load the face recognition model
sfr = SimpleFacerec()
sfr.load_encoding_images("images")

# Open the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Get the current timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Detect known faces
    face_locations, face_names = sfr.detect_known_faces(frame)

    # Draw rectangles and display names on the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Insert into the database
        # Define 'name' and 'timestamp' variables
        db_name = name
        db_timestamp = current_time

        # Insert into the database
        cursor.execute("INSERT INTO RecognitionLog (Name, Timestamp) VALUES (?, ?)", (db_name, db_timestamp))
        conn.commit()

    # Add timestamp to the frame
        cv2.putText(frame, f"Timestamp: {current_time}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop when 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

