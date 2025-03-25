import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np

# Load YOLOv5 model (Replace 'saved_model.pt' with your own model path if needed)
model = YOLO('saved_model.pt')  # Change 'saved_model.pt' to your custom model if necessary

st.title("Hand Detection App")
st.write("This app uses a YOLOv5 model to detect hands in real-time using your webcam.")

# Function for hand detection in real-time using webcam
def detect_hand_from_webcam():
    # Try initializing the webcam (default is 0, might need adjustment)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Unable to access the webcam. Please ensure your device has a working webcam.")
        return

    # Create a placeholder for displaying the video feed
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read frame from webcam.")
            break

        # Convert the frame to RGB (YOLO expects RGB format)
        img = frame[..., ::-1]  # Convert BGR to RGB

        # Inference with YOLOv5 model
        results = model(img)  # Perform inference
        
        # Ensure results is not None and contains boxes
        if results and results[0].boxes:
            boxes = results[0].boxes
            probs = results[0].probs
            names = results[0].names
            for i, box in enumerate(boxes):
                confidence = probs[i] if probs is not None else 0  # Get the confidence score

                # Safely check if `box.cls` exists and is valid
                class_idx = box.cls.item() if box.cls is not None else None
                label = names.get(class_idx, "Unknown") if class_idx is not None else "Unknown"  # Get class label safely

                # Draw the bounding box with enhanced visibility (thicker lines, different color)
                x1, y1, x2, y2 = map(int, box.xywh[0])  # Get the coordinates of the bounding box
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Thicker green box

                # Add label and confidence score to the bounding box
                label_text = f"{label} {confidence:.2f}"  # Label with confidence score
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, label_text, (x1, y1 - 10), font, 0.7, (255, 0, 0), 2)  # Red text

        # Convert the frame to RGB format for Streamlit (it's in BGR format from OpenCV)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update the placeholder with the current frame (real-time video feed)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()

# Run the webcam detection function in Streamlit
if __name__ == "__main__":
    detect_hand_from_webcam()  # Run the Streamlit app without asyncio
