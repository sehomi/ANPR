# ANPR
This repository contains a basic implementation of an Automatic Number Plate Recognition (ANPR). To this end a pretrained YOLOv8n model is responsible for detecting cars and tracking cars. Another YOLOv8n model, responsible for detecting plates, is trained on a [Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) dataset. The training is done on a google colab notebook found [here](https://colab.research.google.com/drive/1ePpRgxyUov99wMeYI_o_ULTrwaED2iZF?usp=sharing). For each detected car, the licence plate is detected and Optical Character Recognition (OCR) is performed to detect digits and numbers. The algorithm is then tested on a sample video with the following output.


## Test
Everything is tested on python 3.10 and CUDA 11.8.
On a fresh python environment, install Pytorch with its [official website](https://pytorch.org/)'s instructions.
Then, install prerequisites with the following command.

```bash
pip install -r requirements.txx
```

Download a sample video such as [this](https://drive.google.com/file/d/1YmHTElM6rh5uBpvaoUYpYTHK2odJkoM6/view) one and place it in the video folder with the specific name of 'sample.mp4'. 
Use the following command to run the code:

```bash
python main.py
```