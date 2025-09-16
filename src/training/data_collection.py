import cv2 as cv


cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If frame reading was unsuccessful, break the loop
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the captured frame
    cv.imshow('EyeCue', frame)

    # Wait for a key press (e.g., 'q' to quit)
    # cv.waitKey(1) waits for 1 millisecond
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()

print("Complete")