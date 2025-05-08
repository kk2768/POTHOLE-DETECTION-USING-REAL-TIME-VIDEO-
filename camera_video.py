import cv2 as cv
import time
import geocoder
import os

# Create output directory if it doesn't exist
result_path = "C:\\Users\\koriv\\Desktop\\pothole-detection-main\\results"
os.makedirs(result_path, exist_ok=True)  # Ensure the results directory exists

# Reading label name from obj.names file
class_name = []
with open(os.path.join("project_files", 'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Importing model weights and config file
net1 = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# Defining the video source
cap = cv.VideoCapture("test.mp4")
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

width = int(cap.get(3))
height = int(cap.get(4))
if width == 0 or height == 0:
    print("Error: Width and height must be greater than zero.")
    exit()

# Initialize VideoWriter
output_file = os.path.join(result_path, 'result.avi')
print(f"Output video path: {output_file}")  # Debugging line
result = cv.VideoWriter(output_file, cv.VideoWriter_fourcc(*'MJPG'), 10, (width, height))

# Initializing parameters
g = geocoder.ip('me')
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
i = 0
b = 0

# Detection loop
while True:
    ret, frame = cap.read()
    frame_counter += 1
    if not ret:
        break
    
    # Analyze the stream with the detection model
    classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        recarea = w * h
        area = width * height
        
        # Drawing detection boxes on frame for detected potholes
        if len(scores) != 0 and scores[0] >= 0.7:
            if (recarea / area) <= 0.1 and box[1] < 600:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv.putText(frame, "%" + str(round(scores[0] * 100, 2)) + " " + label, 
                           (box[0], box[1] - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                
                if i == 0:
                    cv.imwrite(os.path.join(result_path, 'pothole' + str(i) + '.jpg'), frame)
                    with open(os.path.join(result_path, 'pothole' + str(i) + '.txt'), 'w') as f:
                        f.write(str(g.latlng))
                    i += 1
                if i != 0:
                    if (time.time() - b) >= 2:
                        cv.imwrite(os.path.join(result_path, 'pothole' + str(i) + '.jpg'), frame)
                        with open(os.path.join(result_path, 'pothole' + str(i) + '.txt'), 'w') as f:
                            f.write(str(g.latlng))
                        b = time.time()
                        i += 1

    # Writing FPS on frame
    endingTime = time.time() - starting_time
    fps = frame_counter / endingTime
    cv.putText(frame, f'FPS: {fps}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    # Showing and saving result
    cv.imshow('frame', frame)
    result.write(frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# Cleanup
cap.release()
result.release()
cv.destroyAllWindows()
