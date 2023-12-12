
import os
import shutil
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import string

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

def visualize_plate(image, car_box, text):
    
    # The size of the patch to draw the licence plate on.
    h = 80
    w = 400

    # A white image patch with the numbers written on it.
    plate_show = np.ones((h,w, 3), dtype=np.uint8)*255
    cv2.putText(plate_show, text, (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0, 0, 0), 4)

    # Calculating the coordinates on top of the car to attach the licence plate patch.
    x_plate = int((car_box[0][0] + car_box[1][0])/2 - w/2)
    y_plate = int(car_box[0][1] - h)

    image[y_plate:y_plate+h, x_plate:x_plate+w] = plate_show


############################################################################
############# The following post-processing functions    ###################
############# are applied to the ocr output and correct  ###################
############# mis-identified characters for. Taken from: ###################
############# https://github.com/computervisioneng       ###################
############################################################################

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False
    
def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_