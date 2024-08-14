import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Paths to the model files
prototxt_path = 'D:/streamlit/deploy.prototxt'
caffemodel_path = 'D:/streamlit/mobilenet_iter_73000.caffemodel'

# Load the pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Define the class labels MobileNet SSD was trained on
class_labels = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Streamlit App
st.title("Enhanced Object Detection")

# Sidebar
st.sidebar.header("Settings")
option = st.sidebar.selectbox("Choose an Option", ["Webcam Detection", "Image Upload"])

# Create a placeholder for video frames and object counts
stframe = st.empty()
count_placeholder = st.empty()

def detect_objects(frame):
    """Detect objects in the given frame."""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1/255.0, (300, 300), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    object_counts = {label: 0 for label in class_labels[1:]}  # Exclude 'background'

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = class_labels[int(detections[0, 0, i, 1])]

            # Draw bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update object counts
            if label in object_counts:
                object_counts[label] += 1

    return frame, object_counts

if option == "Webcam Detection":
    st.sidebar.write("## Webcam Detection")
    
    if st.sidebar.button("Start Webcam"):
        cap = cv2.VideoCapture(0)  # Open webcam
        
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            st.write("Webcam started. Click 'Stop Webcam' to stop.")
            count_placeholder.write("Detected Objects (Webcam):")

            # Button to stop webcam
            if st.sidebar.button("Stop Webcam"):
                st.write("Stopping webcam...")
                cap.release()
                stframe.empty()
                count_placeholder.empty()
            else:
                # Loop to read frames from the webcam
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Warning: Could not read frame from webcam.")
                        break

                    # Detect objects
                    frame, object_counts = detect_objects(frame)

                    # Convert the image from BGR to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Display the resulting frame in Streamlit
                    stframe.image(frame_rgb, channels="RGB")

                    # Display object counts
                    count_placeholder.write("Detected Objects (Webcam):")
                    for label, count in object_counts.items():
                        if count > 0:
                            count_placeholder.write(f"{label}: {count}")

                # Release the capture
                cap.release()
                stframe.empty()
                count_placeholder.empty()

elif option == "Image Upload":
    st.sidebar.write("## Image Upload")

    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Load the image using PIL
        image = Image.open(uploaded_image)
        image = np.array(image)

        # Detect objects
        image, object_counts = detect_objects(image)

        # Convert the image from BGR to RGB for Streamlit
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the resulting image in Streamlit
        st.image(image_rgb, channels="RGB")

        # Display object counts
        st.write("Detected Objects (Image Upload):")
        for label, count in object_counts.items():
            if count > 0:
                st.write(f"{label}: {count}")
