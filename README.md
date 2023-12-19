# Face Recognition Attendance System 

## Overview
This Python script implements a simple face recognition attendance system using OpenCV and face_recognition. It connects to a SQLite database to manage students' information, track attendance for multiple sessions (in this exemple 5), and log recognition events.

## Features
- Face encoding of students for recognition.
- SQLite database for storing student information and attendance records.
- Automatic determination of the current session based on the day of the week.
- Real-time face recognition from webcam feed.
- Display of recognized students and their attendance status.

## Getting Started
1. Clone the repository to your local machine.
2. Install the required dependencies using: `pip install -r requirements.txt`
3. Fill the folder "images" with images of students for face encoding.
5. Run the script using: `python face_reco.py`

## Database Structure
The SQLite database includes two tables:
1. **Students:**
   - `ID` (INTEGER, Primary Key)
   - `Name` (TEXT, Not Null)
   - `FaceEncoding` (BLOB, Not Null)
   - `Seance_1` to `Seance_5` (INTEGER, Default 0)

2. **RecognitionLog:**
   - `ID` (INTEGER, Primary Key)
   - `StudentID` (INTEGER, Foreign Key referencing Students)
   - `Name` (TEXT, Not Null)
   - `Timestamp` (TEXT, Not Null)
   - `Seance_1` to `Seance_5` (INTEGER, Default 0)

## Usage
1. Run the script and ensure your webcam is connected.
2. The script will automatically recognize faces and mark attendance based on the current session.
3. Press 'Esc' to exit the script.


## License
This project is licensed under the MIT License.

## Acknowledgments
- This project was made for a class of Smart Cities during my third year of computer engineering and IoT studies.

# Blur_known_faces 
is a simpler program that basically does the same thing but instead of marking the attendances `python blur_known_faces.py` starts the webcam and blurs any face detected that is known (and displays the name)
