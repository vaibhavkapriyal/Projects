import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ---------- Core Functions ----------
def preprocess_image(image):
    image = cv2.resize(image, (600, 400))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 13, 15, 15)
    edged = cv2.Canny(blur, 30, 200)
    return image, edged

def find_plate_contour(edged, image):
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            plate_img = image[y:y + h, x:x + w]
            return plate_img, approx
    return None, None

def recognize_plate_text(plate_img):
    gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    config = r'--oem 3 --psm 8'
    text = pytesseract.image_to_string(thresh, config=config)
    return text.strip()

# ---------- GUI Logic ----------
def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        detect_and_display(file_path)

def detect_and_display(path):
    image = cv2.imread(path)
    if image is None:
        result_label.config(text="Error: Unable to read image.")
        return

    processed_image, edged = preprocess_image(image)
    plate_img, plate_contour = find_plate_contour(edged, processed_image)

    if plate_img is not None:
        text = recognize_plate_text(plate_img)
        result_label.config(text=f"Detected Plate: {text}")

        cv2.drawContours(processed_image, [plate_contour], -1, (0, 255, 0), 3)
        show_image(plate_img, plate_canvas)
    else:
        result_label.config(text="No license plate detected.")
        plate_canvas.delete("all")

    show_image(processed_image, original_canvas)

def show_image(cv_image, canvas):
    bgr_to_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(bgr_to_rgb)
    img = ImageTk.PhotoImage(img)

    canvas.image = img
    canvas.create_image(0, 0, anchor=tk.NW, image=img)

# ---------- GUI Layout ----------
root = tk.Tk()
root.title("License Plate Detection & Recognition")
root.geometry("1000x500")
root.configure(bg="#1e1e1e")

tk.Label(root, text="License Plate Detector", font=("Helvetica", 20, "bold"),
         bg="#1e1e1e", fg="#00ffcc").pack(pady=10)

frame = tk.Frame(root, bg="#1e1e1e")
frame.pack()

original_canvas = tk.Canvas(frame, width=600, height=400, bg="black")
original_canvas.grid(row=0, column=0, padx=10)

plate_canvas = tk.Canvas(frame, width=200, height=100, bg="black")
plate_canvas.grid(row=0, column=1, padx=10)

result_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#1e1e1e", fg="white")
result_label.pack(pady=10)

browse_button = tk.Button(root, text="Select Image", command=browse_image,
                          font=("Helvetica", 12), bg="#00ffcc", fg="black", padx=10, pady=5)
browse_button.pack()

root.mainloop()
