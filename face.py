import os
import cv2
from datetime import datetime
from deepface import DeepFace
import mediapipe as mp
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import uuid
from scipy.spatial.distance import cosine

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# EAR threshold for detecting if eyes are open
EYE_AR_THRESHOLD = 0.23

# Indices for left and right eye landmarks (from Mediapipe's face mesh model)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Cosine similarity threshold
COSINE_SIMILARITY_THRESHOLD = 0.8

# Convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Convert base64 to image and save temporarily
def save_base64_image(base64_str):
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    temp_path = "temp_image.jpg"
    img.save(temp_path)
    return temp_path

# Read all base64 strings from a text file
def load_base64_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return [line.strip() for line in file.readlines()]
    return []

# Save base64 string to a text file
def save_to_database(file_path, base64_str):
    with open(file_path, "a") as file:
        file.write(base64_str + "\n")

# Function to calculate eye aspect ratio (EAR)
def calculate_EAR(landmarks, indices):
    A = np.linalg.norm(landmarks[indices[1]] - landmarks[indices[5]])
    B = np.linalg.norm(landmarks[indices[2]] - landmarks[indices[4]])
    C = np.linalg.norm(landmarks[indices[0]] - landmarks[indices[3]])
    EAR = (A + B) / (2.0 * C)
    return EAR

# Extract face embedding using DeepFace
def extract_vector(image_path):
    result = DeepFace.represent(image_path, model_name="VGG-Face", enforce_detection=False)
    return result[0]['embedding']

# Calculate cosine similarity
def calculate_cosine_similarity(vector1, vector2):
    return 1 - cosine(vector1, vector2)

def capture_image():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Image - Press Spacebar to take a photo")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = np.array([[int(p.x * w), int(p.y * h)] for p in face_landmarks.landmark])

                # Calculate the EAR for both eyes
                left_eye_ear = calculate_EAR(landmarks, LEFT_EYE)
                right_eye_ear = calculate_EAR(landmarks, RIGHT_EYE)

                if left_eye_ear > EYE_AR_THRESHOLD and right_eye_ear > EYE_AR_THRESHOLD:
                    cv2.putText(frame, "", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Eyes Closed - Can't capture", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Capture Image - Press Spacebar to take a photo", frame)

        key = cv2.waitKey(1)
        if key % 256 == 32:  # Spacebar to capture
            # Check if the eyes are open before capturing
            if left_eye_ear > EYE_AR_THRESHOLD and right_eye_ear > EYE_AR_THRESHOLD:
                img_name = "captured_image.jpg"
                cv2.imwrite(img_name, frame)
                print(f"Image {img_name} captured!")
                break
            else:
                print("Eyes are closed. Cannot capture the image.")

    cam.release()
    cv2.destroyAllWindows()
    return img_name

def check_spoofing(image_path):
    try:
        face_objs = DeepFace.extract_faces(img_path=image_path, anti_spoofing=True)
        if all(face_obj["is_real"] for face_obj in face_objs):
            return "real"
        else:
            return "spoofed"
    except Exception as e:
        print(f"Error during spoof detection: {str(e)}")
        return "error"

def recognize_face(base64_img, db_folder):
    try:
        # Save the input base64 image to a temporary file
        img_path = save_base64_image(base64_img)
        input_vector = extract_vector(img_path)

        # Iterate through all text files in the database folder
        for file_name in os.listdir(db_folder):
            file_path = os.path.join(db_folder, file_name)

            if os.path.isfile(file_path) and file_name.endswith(".txt"):
                # Load base64 strings from the file
                stored_base64_strings = load_base64_from_file(file_path)

                for stored_base64 in stored_base64_strings:
                    # Extract embedding for the stored image
                    stored_img_path = save_base64_image(stored_base64)
                    stored_vector = extract_vector(stored_img_path)
                    os.remove(stored_img_path)  # Clean up the temporary file

                    # Calculate cosine similarity
                    similarity = calculate_cosine_similarity(input_vector, stored_vector)
                    if similarity >= COSINE_SIMILARITY_THRESHOLD:
                        return True  # Match found
        return False  # No match found
    except Exception as e:
        print(f"Error during face recognition: {str(e)}")
        return False

if __name__ == "__main__":
    db_folder = "database"
    os.makedirs(db_folder, exist_ok=True)  # Create database folder if it doesn't exist

    while True:
        img_path = capture_image()
        spoof_result = check_spoofing(img_path)
        if spoof_result == "spoofed":
            print("Spoof detected! Please retry.")
        elif spoof_result == "real":
            print("Real face detected, proceeding to recognition.")
            img_base64 = image_to_base64(img_path)

            if recognize_face(img_base64, db_folder):
                print("Welcome back! Face recognized.")
                break
            else:
                print("Face not recognized, storing it in the database.")
                unique_file_name = f"{uuid.uuid4()}.txt"
                file_path = os.path.join(db_folder, unique_file_name)
                save_to_database(file_path, img_base64)
                print(f"New face added to the database in {file_path}.")
                break
        else:
            print("Error during spoof detection, please retry.")
