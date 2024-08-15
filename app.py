import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64

# Paths to the model files
prototxt_path = 'deploy.prototxt'
caffemodel_path = 'mobilenet_iter_73000.caffemodel'

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

# Define HTML for webcam capture
webcam_html = """
<script>
    let video;
    let canvas;
    let context;
    let stream;

    function startWebcam() {
        video = document.createElement('video');
        canvas = document.createElement('canvas');
        context = canvas.getContext('2d');
        
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(s) {
                stream = s;
                video.srcObject = stream;
                video.play();
                document.getElementById('webcam-container').appendChild(video);
                requestAnimationFrame(updateFrame);
            })
            .catch(function(err) {
                console.log("Error: " + err);
            });
    }

    function updateFrame() {
        if (video) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            let dataURL = canvas.toDataURL('image/jpeg');
            fetch('/upload_frame', {
                method: 'POST',
                body: JSON.stringify({ image: dataURL }),
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                // Process detection results here
            });
            requestAnimationFrame(updateFrame);
        }
    }

    function stopWebcam() {
        if (stream) {
            let tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
        }
        video.remove();
        canvas.remove();
    }
</script>
<div id="webcam-container">
    <button onclick="startWebcam()">Start Webcam</button>
    <button onclick="stopWebcam()">Stop Webcam</button>
</div>
"""

# Streamlit app layout
st.markdown(webcam_html, unsafe_allow_html=True)

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

# Handle incoming frames
def handle_frame():
    import base64
    import io
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route('/upload_frame', methods=['POST'])
    def upload_frame():
        data = request.json
        image_data = data['image']
        image_data = image_data.split(",")[1]  # Remove the data URL prefix
        image_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_data))
        frame = np.array(image)

        # Detect objects
        frame, object_counts = detect_objects(frame)

        # Convert the image from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', frame_rgb)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'image': jpg_as_text,
            'counts': object_counts
        })

    app.run(port=5000)

if __name__ == "__main__":
    handle_frame()

