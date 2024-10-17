import cv2
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from converter import ReconstructedResNet

# Load the model
model = ReconstructedResNet()  # Use your custom model class here
model.eval()  # Set to evaluation mode
checkpoint = torch.load('checkpoint.ckpt', map_location=torch.device('cpu'))  # Load checkpoint
model.load_state_dict(checkpoint['state_dict'], strict=False)  # Load model weights

# Define preprocessing transformation (depends on the input size your model expects)
preprocess = T.Compose([
    T.Resize((224, 224)),  # Resize the frame to the model's expected size
    T.ToTensor(),          # Convert frame to tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet
])

# Function to run object detection
def detect_objects(frame):
    # Convert the frame (OpenCV format) to a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the image
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(input_tensor)

    # Post-processing: assuming your model returns bounding boxes and class scores
    # You may need to adapt this based on your model's output format
    boxes = outputs[:, :4]  # Assuming the first 4 values are bounding box coordinates
    scores = outputs[:, 4]  # Assuming the fifth value is the score (adjust this based on your model)

    # Draw bounding boxes and labels on the original frame
    for i, box in enumerate(boxes):
        score = scores[i].item()  # Convert the score tensor to a scalar
        if score > 0.5:  # Threshold for confidence
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Object {i+1} - {score:.2f}", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Capture from webcam (0) or load a video/image file
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    detected_frame = detect_objects(frame)

    # Display the frame with detections
    cv2.imshow('Object Detection', detected_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
