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
conn = sqlite3.connect("presence.db")
cursor = conn.cursor()

# Create the Students table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Students (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Name TEXT NOT NULL,
        FaceEncoding BLOB NOT NULL,
        Seance_1 INTEGER DEFAULT 0,
        Seance_2 INTEGER DEFAULT 0,
        Seance_3 INTEGER DEFAULT 0,
        Seance_4 INTEGER DEFAULT 0,
        Seance_5 INTEGER DEFAULT 0
    )
''')

# Create the RecognitionLog table with a foreign key constraint
cursor.execute('''
    CREATE TABLE IF NOT EXISTS RecognitionLog (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        StudentID INTEGER,
        Name TEXT NOT NULL,
        Timestamp TEXT NOT NULL,
        Seance_1 INTEGER DEFAULT 0,
        Seance_2 INTEGER DEFAULT 0,
        Seance_3 INTEGER DEFAULT 0,
        Seance_4 INTEGER DEFAULT 0,
        Seance_5 INTEGER DEFAULT 0,
        FOREIGN KEY (StudentID) REFERENCES Students (ID)
    )
''')

# Commit changes
conn.commit()

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_student_ids = []
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*"))
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract face encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Get student name from the filename (assuming filenames are student names)
            name = os.path.splitext(os.path.basename(img_path))[0]

            # Check if the student already exists in the Students table based on name and face encoding
            cursor.execute("SELECT ID FROM Students WHERE Name = ? AND FaceEncoding = ?", (name, img_encoding.tobytes()))
            existing_student = cursor.fetchone()

            if existing_student:
                # If the student exists, use the existing ID
                student_id = existing_student[0]
            else:
                # If the student doesn't exist, insert a new record
                cursor.execute("INSERT INTO Students (Name, FaceEncoding) VALUES (?, ?)", (name, img_encoding.tobytes()))
                conn.commit()
                student_id = cursor.lastrowid

            # Keep track of the student IDs
            self.known_face_encodings.append(img_encoding)
            self.known_student_ids.append(student_id)

        print("Encoding images loaded")

    def determine_current_seance(self):
        # Get the current day of the week (Monday is 0 and Sunday is 6)
        current_day = datetime.date.today().weekday()

        # Map days to seances
        seance_mapping = {
            0: 1,  # Monday corresponds to Seance 1
            1: 2,  # Tuesday corresponds to Seance 2
            2: 3,  # Wednesday corresponds to Seance 3
            3: 4,  # Thursday corresponds to Seance 4
            4: 5,  # Friday corresponds to Seance 5
        }

        # Determine the current seance based on the current day
        current_seance = seance_mapping.get(current_day, 0)  # Default to 0 if no mapping found

        # Check if there are any previous attendance records for the current week       
        for seance in range(1, current_seance + 1):
            column_name = f"Seance_{seance}"
            cursor.execute(f"SELECT COUNT(*) FROM RecognitionLog WHERE {column_name} = 1")
            attendance_count = cursor.fetchone()[0]

            if attendance_count == 0:
                # If no attendance records found for the current seance, break the loop
                break

        return current_seance

    def update_seance_in_students_table(self, student_id, seance):
        # Update the corresponding Seance_X column in the Students table

        # Build the column name based on the seance number
        seance_column_name = f"Seance_{seance}"

        # Check if the column exists in the Students table
        cursor.execute(f"PRAGMA table_info(Students);")
        columns = cursor.fetchall()
        seance_column_exists = any(column[1] == seance_column_name for column in columns)

        if not seance_column_exists:
            # If the column doesn't exist, add it to the Students table
            cursor.execute(f"ALTER TABLE Students ADD COLUMN {seance_column_name} INTEGER DEFAULT 0;")
            conn.commit()

        # Update the Seance_X column in the Students table
        cursor.execute(f"UPDATE Students SET {seance_column_name} = 1 WHERE ID = ?", (student_id,))
        conn.commit()

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
                # Retrieve student ID and name
                student_id = self.known_student_ids[best_match_index]
                student_name = cursor.execute("SELECT Name FROM Students WHERE ID = ?", (student_id,)).fetchone()[0]

                # Determine the current seance
                current_seance = self.determine_current_seance()

                # Mark attendance based on the detected seance
                seance_attendance_column = f"Seance_{current_seance}"
                cursor.execute(f"""
                    INSERT INTO RecognitionLog (StudentID, Name, Timestamp, {seance_attendance_column})
                    VALUES (?, ?, ?, 1)
                """, (student_id, student_name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()

                # Update the corresponding Seance_X column in the Students table
                self.update_seance_in_students_table(student_id, current_seance)

                name = student_name

            face_names.append(name)

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing

        return face_locations.astype(int), face_names

# Load the face recognition model
sfr = SimpleFacerec()
sfr.load_encoding_images("images")

# Open the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Detect known faces
    face_locations, face_names = sfr.detect_known_faces(frame)

    # Draw rectangles and display names on the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop when 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
