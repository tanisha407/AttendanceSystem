
import cv2
import os
import numpy as np
import datetime
import time
import csv
from deepface import DeepFace
from tabulate import tabulate

# Load Haar Cascade once globally
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ----------------------------- Step 1: Capture face images -----------------------------
def capture_images(name, num_images):
    directory = "captured_images"
    os.makedirs(directory, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print("\n[INFO] Press 'c' to capture images or 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        cv2.imshow('Capture - Press "c" to Capture, "q" to Quit', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                file_path = os.path.join(directory, f"{name}_{count}.png")
                cv2.imwrite(file_path, face)
                print(f"[INFO] Saved: {file_path}")
                count += 1

                if count >= num_images:
                    break

        elif key & 0xFF == ord('q'):
            break

        if count >= num_images:
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------- Step 2: Load data for training -----------------------------
def get_images_and_labels(directory):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    for file in os.listdir(directory):
        if file.endswith(".png"):
            path = os.path.join(directory, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            name = file.split("_")[0]

            if name not in label_dict:
                label_dict[name] = current_label
                current_label += 1

            images.append(img)
            labels.append(label_dict[name])

    return images, np.array(labels), label_dict

# ----------------------------- Step 3: Train face recognizer -----------------------------
def train_recognizer(directory):
    images, labels, label_dict = get_images_and_labels(directory)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)

    recognizer.save("face_recognizer_model.yml")
    np.save("label_dict.npy", label_dict)

    print("[INFO] Model and labels saved successfully.")

# ----------------------------- Step 4: Face Recognition and Attendance -----------------------------
def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_recognizer_model.yml")
    label_dict = np.load("label_dict.npy", allow_pickle=True).item()
    reverse_dict = {v: k for k, v in label_dict.items()}

    cap = cv2.VideoCapture(0)
    attended = set()

    print("\n[INFO] Face Recognition Running. Smile to mark attendance ")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera not working.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(roi)
            name = reverse_dict.get(label, "Unknown")

            try:
                result = DeepFace.analyze(frame[y:y+h, x:x+w], actions=['emotion'], enforce_detection=False)
                if isinstance(result, list): result = result[0]
                emotion = result.get("dominant_emotion", None)
            except Exception as e:
                print(f"[ERROR] Emotion detection failed: {e}")
                emotion = None

            # Mark attendance only when happy
            if name not in attended and emotion == "happy":
                timestamp = datetime.datetime.now()
                date = timestamp.strftime("%d-%m-%Y")
                time_str = timestamp.strftime("%H:%M:%S")

                with open("attendance_log.csv", "a", newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([date, time_str, name])

                attended.add(name)
                print(f"[INFO] Attendance marked for {name} ")

                cv2.putText(frame, f"{name} ({emotion})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Face Recognition - Press 'q' to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Attendance recording finished.")

# ----------------------------- Step 5: View Attendance Log -----------------------------
def view_attendance():
    file_path = "attendance_log.csv"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader, None)
            rows = list(reader)

            if headers:
                print(tabulate(rows, headers=headers, tablefmt='grid'))
            else:
                print("[INFO] No data in the attendance log.")
    else:
        print("[INFO] No attendance file found yet.")

# ----------------------------- Main Menu -----------------------------
def main():
    while True:
        if os.path.exists("face_recognizer_model.yml") and os.path.exists("label_dict.npy"):
            print("\n--- Attendance System ---")
            print("1. Register New User")
            print("2. Mark Attendance")
            print("3. View Attendance")
            print("4. Exit")

            choice = input("Choose an option: ").strip()

            if choice == '1':
                name = input("Enter Name: ")
                count = int(input("How many images to capture?: "))
                capture_images(name, count)
                train_recognizer("captured_images")

            elif choice == '2':
                recognize_faces()

            elif choice == '3':
                view_attendance()

            elif choice == '4':
                print("[EXIT] Goodbye!")
                break

            else:
                print("[ERROR] Invalid input.")

        else:
            print("\n[INFO] No trained data found. Register new user first.")
            name = input("Enter Name: ")
            count = int(input("How many images to capture?: "))
            capture_images(name, count)
            train_recognizer("captured_images")

if __name__ == "__main__":
    main()
import datetime
import time
import csv
from deepface import DeepFace
from tabulate import tabulate

# Load Haar Cascade once globally
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# ----------------------------- Step 1: Capture face images -----------------------------

def capture_images(name, num_images):
    directory = "captured_images"
    os.makedirs(directory, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print("\n[INFO] Press 'c' to capture images or 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        cv2.imshow('Capture - Press "c" to Capture, "q" to Quit', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                file_path = os.path.join(directory, f"{name}_{count}.png")
                cv2.imwrite(file_path, face)
                print(f"[INFO] Saved: {file_path}")
                count += 1

                if count >= num_images:
                    break

        elif key & 0xFF == ord('q'):
            break

        if count >= num_images:
            break

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------- Step 2: Load data for training -----------------------------

def get_images_and_labels(directory):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    for file in os.listdir(directory):
        if file.endswith(".png"):
            path = os.path.join(directory, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            name = file.split("_")[0]

            if name not in label_dict:
                label_dict[name] = current_label
                current_label += 1

            images.append(img)
            labels.append(label_dict[name])

    return images, np.array(labels), label_dict


# ----------------------------- Step 3: Train face recognizer -----------------------------

def train_recognizer(directory):
    images, labels, label_dict = get_images_and_labels(directory)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)

    recognizer.save("face_recognizer_model.yml")
    np.save("label_dict.npy", label_dict)

    print("[INFO] Model and labels saved successfully.")


# ----------------------------- Step 4: Face Recognition and Attendance -----------------------------

def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_recognizer_model.yml")
    label_dict = np.load("label_dict.npy", allow_pickle=True).item()
    reverse_dict = {v: k for k, v in label_dict.items()}

    cap = cv2.VideoCapture(0)
    attended = set()

    print("\n[INFO] Face Recognition Running. Smile to mark attendance 😊")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera not working.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(roi)
            name = reverse_dict.get(label, "Unknown")

            try:
                result = DeepFace.analyze(frame[y:y+h, x:x+w], actions=['emotion'], enforce_detection=False)
                if isinstance(result, list): result = result[0]
                emotion = result.get("dominant_emotion", None)
            except Exception as e:
                print(f"[ERROR] Emotion detection failed: {e}")
                emotion = None

            # Mark attendance only when happy
            if name not in attended and emotion == "happy":
                timestamp = datetime.datetime.now()
                date = timestamp.strftime("%d-%m-%Y")
                time_str = timestamp.strftime("%H:%M:%S")

                with open("attendance_log.csv", "a", newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([date, time_str, name])

                attended.add(name)
                print(f"[INFO] Attendance marked for {name} 😊")

                cv2.putText(frame, f"{name} ({emotion})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Face Recognition - Press 'q' to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Attendance recording finished.")


# ----------------------------- Step 5: View Attendance Log -----------------------------

def view_attendance():
    file_path = "attendance_log.csv"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader, None)
            rows = list(reader)

            if headers:
                print(tabulate(rows, headers=headers, tablefmt='grid'))
            else:
                print("[INFO] No data in the attendance log.")
    else:
        print("[INFO] No attendance file found yet.")


# ----------------------------- Main Menu -----------------------------

def main():
    while True:
        if os.path.exists("face_recognizer_model.yml") and os.path.exists("label_dict.npy"):
            print("\n--- Attendance System ---")
            print("1. Register New User")
            print("2. Mark Attendance")
            print("3. View Attendance")
            print("4. Exit")

            choice = input("Choose an option: ").strip()

            if choice == '1':
                name = input("Enter Name: ")
                count = int(input("How many images to capture?: "))
                capture_images(name, count)
                train_recognizer("captured_images")

            elif choice == '2':
                recognize_faces()

            elif choice == '3':
                view_attendance()

            elif choice == '4':
                print("[EXIT] Goodbye!")
                break

            else:
                print("[ERROR] Invalid input.")

        else:
            print("\n[INFO] No trained data found. Register new user first.")
            name = input("Enter Name: ")
            count = int(input("How many images to capture?: "))
            capture_images(name, count)
            train_recognizer("captured_images")


if __name__ == "__main__":
    main()