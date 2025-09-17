import cv2 as cv
from datetime import datetime
import os

CAPTURE_COUNT = 20

root_dir = os.getcwd()
data_object_path = os.path.join(root_dir, "data/train/object")
data_no_object_path = os.path.join(root_dir, "data/train/no_object")
os.makedirs(data_object_path, exist_ok=True)
os.makedirs(data_no_object_path, exist_ok=True)


cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


Capture_Data = 0
frame_count = 0
while Capture_Data < CAPTURE_COUNT: 
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)

    # If frame reading was unsuccessful, break the loop
    if not ret:
        print("Error: Could not read frame.")
        break
    
    
    # Display the captured frame
    cv.imshow(f"EyeCue-Data Collector", frame)
    

    #Create file path and timestamp .jpg to send to data folder 
    ts = int(datetime.now().timestamp())
    
    
    # Don't capture everyframe. 30Hz camera. Every 30th frame. ~ 1 image/sec
    if frame_count % 30 == 0 and Capture_Data < CAPTURE_COUNT/2:
        cv.setWindowTitle("EyeCue-Data Collector", "Detect Object")
        file_path = os.path.join(data_object_path,f"{ts}.jpg")
        cv.imwrite(file_path, frame)
        Capture_Data += 1
       
    if frame_count % 30 == 0 and Capture_Data >= CAPTURE_COUNT/2: 
        cv.setWindowTitle("EyeCue-Data Collector", "Detect NO Object")
        file_path = os.path.join(data_no_object_path,f"{ts}.jpg")
        cv.imwrite(file_path, frame)
        Capture_Data += 1
            


    # Wait for a key press (e.g., 'q' to quit)
    # cv.waitKey(1) waits for 1 millisecond
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count+=1

    print(Capture_Data)

cap.release()
cv.destroyAllWindows()

print("Complete")