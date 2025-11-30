# add_negatives.py
# This script takes background images (no tennis ball)
# and copies them into data/images/all, and creates empty label files
# in data/labels/all. Empty txt = no object.

from pathlib import Path
import shutil

def add_background_folder(source_folder, images_out_folder, labels_out_folder):
    source_folder = Path(source_folder)
    images_out_folder = Path(images_out_folder)
    labels_out_folder = Path(labels_out_folder)

    images_out_folder.mkdir(parents=True, exist_ok=True)
    labels_out_folder.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(source_folder.glob(ext))

    print("Background images in", source_folder, ":", len(image_paths))

    for img_path in image_paths:
        # copy image
        out_img_path = images_out_folder / img_path.name
        if not out_img_path.exists():
            shutil.copy(img_path, out_img_path)

        # create empty label file
        label_path = labels_out_folder / (img_path.stem + ".txt")
        if not label_path.exists():
            label_path.write_text("")
        print("[NEGATIVE]", img_path.name, "->", label_path.name)


def main():
    images_out = "data/images/all"
    labels_out = "data/labels/all"

    # self background
    add_background_folder("data/raw/self_background", images_out, labels_out)

    # supplemental background
    add_background_folder("data/raw/supp_background", images_out, labels_out)

    print("Finished adding negative (no-ball) images.")

if __name__ == "__main__":
    main()
