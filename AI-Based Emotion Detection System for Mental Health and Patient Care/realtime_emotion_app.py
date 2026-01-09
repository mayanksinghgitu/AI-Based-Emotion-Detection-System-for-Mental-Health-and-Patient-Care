import cv2
import numpy as np
import tensorflow as tf
import time
import os
import webbrowser
import threading
import tkinter as tk
from tkinter import messagebox

# =========================
# CONFIG
# =========================

#MODEL_PATH = r"C:\Users\mayan\Desktop\Artificial_Intelligence_DS_ML\emotion_model_realworld.h5"
MODEL_PATH = r"C:\Users\mayan\Desktop\Artificial_Intelligence_DS_ML\emotion_model_realworld2.h5"

CASCADE_PATH = r"C:\Users\mayan\Desktop\Artificial_Intelligence_DS_ML\Deep_Learning\haarcascade_frontalface_default.xml"

# NOTE: make sure this order matches your training class order:
# CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

IMG_SIZE = 100   # <-- UPDATED to match training image size (100x100)
CONFIDENCE_THRESHOLD = 0.30

TREATMENT_URL = "file:///C:/Users/mayan/Desktop/Artificial_Intelligence_DS_ML/Deep_Learning/index.html"

# Predefined colors (BGR)
EMOTION_COLORS = {
    "Angry": (0, 0, 255),
    "Disgust": (0, 128, 0),
    "Fear": (128, 0, 128),
    "Happy": (0, 255, 255),
    "Sad": (255, 0, 0),
    "Surprise": (255, 255, 0),
    "Neutral": (200, 200, 200),
    "Unknown": (0, 255, 0)
}

# Global control flag (for GUI <-> video loop)
running = True
cap = None


# =========================
# UTILS
# =========================

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"[INFO] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("[INFO] Model loaded successfully.")
    return model


def load_face_cascade(cascade_path):
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar cascade file not found: {cascade_path}")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise IOError("Error loading Haar cascade. File might be corrupted.")
    print("[INFO] Haar cascade loaded successfully.")
    return face_cascade


def preprocess_face(gray_face):
    # Resize to 100x100 to match your training input
    face_resized = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
    face_resized = face_resized.astype("float32") / 255.0  # must match training
    face_resized = np.expand_dims(face_resized, axis=-1)   # (100,100,1)
    return face_resized


def get_color_for_emotion(label):
    return EMOTION_COLORS.get(label, EMOTION_COLORS["Unknown"])


# =========================
# VIDEO LOOP (runs in thread)
# =========================

def video_loop(model, face_cascade):
    global running, cap

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        running = False
        return

    # Smaller resolution for smoother FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    fps = 0.0

    print("[INFO] Press 'q' in video window OR use Exit button.")

    window_name = "Facial Emotion Recognition"

    while running:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # FPS calc
        current_time = time.time()
        dt = current_time - prev_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        prev_time = current_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60)
        )

        face_imgs = []
        face_coords = []

        for (x, y, w, h) in faces:
            face_gray = gray[y:y + h, x:x + w]
            processed = preprocess_face(face_gray)
            face_imgs.append(processed)
            face_coords.append((x, y, w, h))

        if len(face_imgs) > 0:
            batch = np.stack(face_imgs, axis=0)  # (N,100,100,1)
            preds = model.predict(batch, verbose=0)

            for i, (x, y, w, h) in enumerate(face_coords):
                probs = preds[i]
                label_idx = int(np.argmax(probs))
                confidence = float(np.max(probs))

                if confidence < CONFIDENCE_THRESHOLD:
                    emotion = "Unknown"
                else:
                    emotion = EMOTION_LABELS.get(label_idx, "Unknown")

                color = get_color_for_emotion(emotion)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = f"{emotion} ({confidence*100:.1f}%)"
                cv2.putText(
                    frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2, cv2.LINE_AA
                )

        # FPS text
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(window_name, frame)

        # Allow 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False
            break

        # If user closes OpenCV window with [X], stop loop cleanly
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            running = False
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Video loop stopped.")


# =========================
# GUI CALLBACKS
# =========================

def on_exit(root):
    """Exit button handler."""
    global running, cap
    running = False
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    root.destroy()
    print("[INFO] Application closed by user.")


def on_about():
    """Show About dialog."""
    message = (
        "Facial Emotion Recognition System\n"
        "---------------------------------\n"
        "• Real-time emotion detection using Mini-Xception model\n"
        "• Built with TensorFlow, Keras & OpenCV\n\n"
        "Developer: Mayank Singh\n"
        "Tech: Python, Deep Learning, Computer Vision"
    )
    messagebox.showinfo("About", message)


def on_treatment():
    """Open treatment suggestion web page."""
    webbrowser.open(TREATMENT_URL)


# =========================
# MAIN
# =========================

def main():
    global running

    # TF GPU memory growth (optional)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("[WARN] Could not set memory growth:", e)

    model = load_model(MODEL_PATH)
    face_cascade = load_face_cascade(CASCADE_PATH)

    # -------- Tkinter GUI --------
    root = tk.Tk()
    root.title("Emotion Recognition Control Panel")
    root.geometry("420x220")
    root.resizable(False, False)

    # Dark-ish background for nicer look
    root.configure(bg="#222222")

    title_label = tk.Label(
        root,
        text="Facial Emotion Recognition",
        font=("Segoe UI", 16, "bold"),
        fg="white",
        bg="#222222"
    )
    title_label.pack(pady=10)

    subtitle_label = tk.Label(
        root,
        text="Webcam-based real-time emotion detection",
        font=("Segoe UI", 10),
        fg="#cccccc",
        bg="#222222"
    )
    subtitle_label.pack(pady=(0, 15))

    # Frame for buttons
    btn_frame = tk.Frame(root, bg="#222222")
    btn_frame.pack(pady=10)

    # About button
    about_btn = tk.Button(
        btn_frame,
        text="About",
        font=("Segoe UI", 11),
        width=16,
        command=on_about,
        bg="#444444",
        fg="white",
        activebackground="#555555",
        activeforeground="white",
        relief="groove",
        bd=2
    )
    about_btn.grid(row=0, column=0, padx=10, pady=5)

    # Treatment suggestion button
    treatment_btn = tk.Button(
        btn_frame,
        text="Treatment Suggestion",
        font=("Segoe UI", 11),
        width=18,
        command=on_treatment,
        bg="#0078D7",
        fg="white",
        activebackground="#0063b1",
        activeforeground="white",
        relief="groove",
        bd=2
    )
    treatment_btn.grid(row=0, column=1, padx=10, pady=5)

    # Exit button (RED)
    exit_btn = tk.Button(
        root,
        text="Exit",
        font=("Segoe UI", 12, "bold"),
        width=20,
        command=lambda: on_exit(root),
        bg="#cc0000",
        fg="white",
        activebackground="#990000",
        activeforeground="white",
        relief="raised",
        bd=3
    )
    exit_btn.pack(pady=15)

    # Start video loop in a separate thread
    running = True
    video_thread = threading.Thread(
        target=video_loop, args=(model, face_cascade), daemon=True
    )
    video_thread.start()

    # Start GUI loop
    root.mainloop()

    # When GUI closes, ensure everything stops
    running = False
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
