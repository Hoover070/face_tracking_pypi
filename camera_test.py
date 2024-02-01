import cv2
import numpy as np

# Load the pre-trained model and weights (adjust paths as necessary)
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')

# Open video
video_capture = cv2.VideoCapture('videos/test_1.mp4')

print("Network loaded:", net.empty())

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Prepare the frame for processing
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Detect faces
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections and draw boxes for each face
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the video
video_capture.release()
cv2.destroyAllWindows()