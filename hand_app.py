import cv2
import streamlit as st
from ultralytics import YOLO
import asyncio

# Load YOLOv5 model (Replace 'saved_model.pt' with your own model path if needed)
model = YOLO('saved_model.pt')  # Change 'saved_model.pt' to your custom model if necessary

st.title("Hand Detection App")
st.write("This app uses a YOLOv5 model to detect hands in real-time using your webcam.")


import torch
import torch.nn as nn

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ImageNet_lightweight(nn.Module):
    def __init__(self, C=25):
        super(ImageNet_lightweight, self).__init__()
        self.BN1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=1, stride = 1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2, 2),
        )
        self.BN2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            ResidualBlock(256, 256),
            #nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, C)
        )
    def forward(self, x):
            x = self.BN1(x)
            #x= self.BN2(x)
            #print(x.shape)
            x = self.fc(x)
            return x
model_c = torch.load("hand_sign_classification_model.pth")
model_c.eval()
dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 
 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 
 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}

# Function for hand detection in real-time using webcam
def detect_hand_from_webcam():
    # Initialize the webcam (default is 0, might need adjustment)
    cap = cv2.VideoCapture(0)
    label_placeholder = st.empty()
    # Create a placeholder for displaying the video feed
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
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
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the coordinates of the bounding box #use xyxy, not xywh
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Thicker green box

                # Add label and confidence score to the bounding box
                label_text = f"{label} {confidence:.2f}"  # Label with confidence score
                font = cv2.FONT_HERSHEY_SIMPLEX
                cropped_hand = frame[y1:y2, x1:x2]
                #print(y1,y2,x1,x2)
                if cropped_hand.size > 0 and x2 > x1 and y2 > y1:
                    gray_hand = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2GRAY)
                    resized_hand = cv2.resize(gray_hand, (28, 28))
                    hand = torch.tensor(resized_hand, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        pred = model_c(hand)
                        predicted_class = torch.argmax(pred).item()
                    pred_char = dict.get(predicted_class, "Invalid number")
                    label_text = f"{label} {confidence:.2f} | Class: {pred_char}"
                    label_placeholder.markdown(f"### Predicted Label: {pred_char}")
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, label_text, (x1, y1 - 10), font, 0.7, (255, 0, 0), 2)
                #cv2.putText(frame, label_text, (x1, y1 - 10), font, 0.7, (255, 0, 0), 2)  # Red text

        # Convert the frame to RGB format for Streamlit (it's in BGR format from OpenCV)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update the placeholder with the current frame (real-time video feed)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)


    cap.release()

# Async wrapper to run the Streamlit function
async def run():
    detect_hand_from_webcam()

# Run the webcam detection function in Streamlit within an asyncio event loop
if __name__ == "__main__":
    asyncio.run(run())  # Run the Streamlit app within an event loop
