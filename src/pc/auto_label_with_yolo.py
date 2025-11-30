# auto_label_with_yolo.py
# This script uses YOLOv8 (pretrained) to find tennis balls in images
# and saves YOLO txt label files for them.
# I use it for self_ball and supp_ball folders.

from ultralytics import YOLO
from pathlib import Path

def main():
    # where my raw ball images are
    self_ball_folder = Path("data/raw/self_ball")
    supp_ball_folder = Path("data/raw/supp_ball")

    # where I want all images and labels to go
    images_out_folder = Path("data/images/all")
    labels_out_folder = Path("data/labels/all")

    images_out_folder.mkdir(parents=True, exist_ok=True)
    labels_out_folder.mkdir(parents=True, exist_ok=True)

    # load tiny YOLOv8 model (COCO pretrained, has "sports ball" class)
    model = YOLO("yolov8n.pt")

    # I will run the same function on both self and supplemental balls
    auto_label_folder(self_ball_folder, images_out_folder, labels_out_folder, model)
    auto_label_folder(supp_ball_folder, images_out_folder, labels_out_folder, model)

    print("Done auto-labeling ball images with YOLO.")


def auto_label_folder(input_folder, images_out_folder, labels_out_folder, model):
    print(f"Running YOLO on folder: {input_folder}")

    # collect all jpg and png files
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(input_folder.glob(ext))

    print("Number of images found:", len(image_paths))

    # find class id for "sports ball" in YOLO's labels
    sports_ball_ids = []
    for class_id, name in model.names.items():
        if "sports ball" in name.lower():
            sports_ball_ids.append(class_id)

    if len(sports_ball_ids) == 0:
        print("Error: could not find 'sports ball' class in YOLO model.")
        return

    print("Using YOLO class ids for sports ball:", sports_ball_ids)

    for img_path in image_paths:
        # run YOLO prediction on the image
        results = model.predict(
            source=str(img_path),
            imgsz=640,
            conf=0.5,
            verbose=False
        )

        r = results[0]
        boxes = r.boxes

        # keep only sports ball detections
        ball_boxes = []
        for b in boxes:
            cls_id = int(b.cls.item())
            if cls_id in sports_ball_ids:
                ball_boxes.append(b)

        if len(ball_boxes) == 0:
            # YOLO didn't find any ball
            print("[NO BALL DETECTED]", img_path.name)
            continue

        # copy or save image into images_out_folder
        out_img_path = images_out_folder / img_path.name
        if not out_img_path.exists():
            out_img_path.write_bytes(img_path.read_bytes())

        # write YOLO label file for this image (class 0 = tennis_ball)
        h, w = r.orig_shape  # height, width
        label_path = labels_out_folder / (img_path.stem + ".txt")

        lines = []
        for b in ball_boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            # YOLO format: class cx cy w h
            line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            lines.append(line)

        label_path.write_text("\n".join(lines))
        print("[LABELED]", img_path.name, "->", label_path.name)


if __name__ == "__main__":
    main()
