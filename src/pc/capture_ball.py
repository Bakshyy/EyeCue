# capture_ball_300.py
# This script captures 300 images of a tennis ball using my webcam.
# The images will be saved to data/raw/self_ball.

import cv2
from pathlib import Path
import time

# number of images I want to capture each time I run the script
NUM_IMAGES = 300

# where to save the images
OUTPUT_FOLDER = Path("data/raw/self_ball")

def get_start_index(folder):
    """Find the next available number for saving new images."""
    folder.mkdir(parents=True, exist_ok=True)
    numbers = []

    for img in folder.glob("*.jpg"):
        name = img.stem  # example: "self_ball_00023"
        parts = name.split("_")
        if parts[-1].isdigit():
            numbers.append(int(parts[-1]))

    if len(numbers) == 0:
        return 1
    else:
        return max(numbers) + 1


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # open webcam (0 means default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    next_number = get_start_index(OUTPUT_FOLDER)
    saved_count = 0

    print("Starting BALL image capture...")
    print("Press 'q' to stop early.")

    while saved_count < NUM_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame.")
            break

        # smaller preview window so it's not huge
        preview = cv2.resize(frame, (960, 540))

        # show some text on the preview window
        cv2.putText(preview, f"BALL images saved: {saved_count}/{NUM_IMAGES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        cv2.imshow("Ball Capture", preview)

        # save every frame (automatic capture)
        filename = f"self_ball_{next_number:05d}.jpg"
        cv2.imwrite(str(OUTPUT_FOLDER / filename), frame)

        print("Saved:", filename)

        saved_count += 1
        next_number += 1

        # let me quit early if needed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Stopped early by user.")
            break

        # small delay so I have time to move the ball around
        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()
    print("Ball image capture complete!")


if __name__ == "__main__":
    main()
