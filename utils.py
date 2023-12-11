
import os
import shutil
import xml.etree.ElementTree as ET
import cv2

def xml_to_yolo(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

def convert_dataset(dir, training_percentage=0.8):

    # Count files in dataset. Find maximum number for training.
    files = [f for f in os.listdir(f"{dir}/annotations") if os.path.isfile(os.path.join(f"{dir}/annotations", f))]
    num_files = len(files)
    max_num_training = int(training_percentage*num_files)

    # Create train folder. If it exists, remove it then recreate it.
    if not os.path.exists(f"{dir}/train/labels/"):
        os.makedirs(f"{dir}/train/labels/")
        os.makedirs(f"{dir}/train/images/")
    else:
        shutil.rmtree(f"{dir}/train/")
        os.makedirs(f"{dir}/train/labels/")
        os.makedirs(f"{dir}/train/images/")

    # Create validation folder. If it exists, remove it then recreate it.
    if not os.path.exists(f"{dir}/validation/labels/"):
        os.makedirs(f"{dir}/validation/labels/")
        os.makedirs(f"{dir}/validation/images/")
    else:
        shutil.rmtree(f"{dir}/validation/")
        os.makedirs(f"{dir}/validation/labels/")
        os.makedirs(f"{dir}/validation/images/")

    # Convert from kaggle xml format to YOLO format.
    training_count = 0
    for filename in os.listdir(f"{dir}/annotations"):

        tree = ET.parse(f"{dir}/annotations/{filename}")
        root = tree.getroot()
        img_name = root.find("filename").text
        name = root.find("filename").text.replace(".png", "")
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)

        for obj in root.findall('object'):
            box = []        
            for x in obj.find("bndbox"):
                box.append(int(x.text))

            yolo_box = xml_to_yolo(box, width, height)
            line = f"0 {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}"

            # Split training, validation
            if training_count < max_num_training:
                shutil.copy(f"{dir}/images/{img_name}", f"{dir}/train/images/")
                with open(f"{dir}/train/labels/{name}.txt", "a") as file:
                    file.write(f"{line}\n")
            else:
                shutil.copy(f"{dir}/images/{img_name}", f"{dir}/validation/images/")
                with open(f"{dir}/validation/labels/{name}.txt", "a") as file:
                    file.write(f"{line}\n")

            training_count += 1

def visualize_plate(image, data):

    x1, y1, x2, y2 = data['box']
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    if 'licence_plate' in data:
        box, text, _ = data['licence_plate']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        # cv2.rectangle(cropped_img, (int(x1_plate), int(y1_plate)), (int(x2_plate), int(y2_plate)), (255, 0, 0), 2)
        text_x = int(x1)
        text_y = int(y1) - 10
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)