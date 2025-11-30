# This script captures 300 background images (no tennis ball).
# The images are saved to data/raw/self_background.

import cv2
from pathlib import Path
import time

NUM_IMAGES = 300
OUTPUT_FOLDER = Path("data/raw/self_background")

def get_start_index(folder):
    """Find next available index for naming background images."""
    folder.mkdir(parents=True, exist_ok=True)
    numbers = []

    for img in folder.glob("*.jpg"):
        name = img.stem  # ex: "self_background_00014"
        parts = name.split("_")
        if parts[-1].isdigit():
            numbers.append(int(parts[-1]))

    if len(numbers) == 0:
        return 1
    else:
        return max(numbers) + 1


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not access webcam.")
        return

    next_number = get_start_index(OUTPUT_FOLDER)
    saved = 0

    print("Starting BACKGROUND image capture...")
    print("Press 'q' to stop early.")

    while saved < NUM_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print("Warning: could not read frame.")
            break

        preview = cv2.resize(frame, (960, 540))
        cv2.putText(preview, f"Background saved: {saved}/{NUM_IMAGES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        cv2.imshow("Background Capture", preview)

        # save automatically each loop
        filename = f"self_background_{next_number:05d}.jpg"
        cv2.imwrite(str(OUTPUT_FOLDER / filename), frame)

        print("Saved:", filename)

        saved += 1
        next_number += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Stopped early by user.")
            break

        # slight delay so the scene changes between captures
        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()
    print("Background image capture complete!")


if __name__ == "__main__":
    main()
