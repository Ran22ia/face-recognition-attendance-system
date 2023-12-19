import os 
import cv2
import face_recognition

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Relative paths to the images in the "images" folder
img_path1 = os.path.join(current_directory, "images", "Person1.jpg")
img_path2 = os.path.join(current_directory, "images", "Person2.jpg")

# Load images
img1 = cv2.imread(img_path1)
img2 = cv2.imread(img_path2)

if img1 is None or img2 is None:
    print("Error: Unable to read the image file.")
else:
    # Resize images to a smaller size
    resized_img1 = cv2.resize(img1, (800, 600))
    resized_img2 = cv2.resize(img2, (800, 600))

    # Convert images to RGB format
    rgb_img1 = cv2.cvtColor(resized_img1, cv2.COLOR_BGR2RGB)
    rgb_img2 = cv2.cvtColor(resized_img2, cv2.COLOR_BGR2RGB)

    # Get face encodings
    face_encodings1 = face_recognition.face_encodings(rgb_img1)
    face_encodings2 = face_recognition.face_encodings(rgb_img2)

    if not face_encodings1:
        print("No face found in the first image.")
    elif not face_encodings2:
        print("No face found in the second image.")
    else:
        # Use the first face in each image for comparison
        img_encoding1 = face_encodings1[0]
        img_encoding2 = face_encodings2[0]

        # Compare faces
        result = face_recognition.compare_faces([img_encoding1], img_encoding2)
        print("Result: ", result)

        # Display resized images
        cv2.imshow("Img 1", resized_img1)
        cv2.imshow("Img 2", resized_img2)
        cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

