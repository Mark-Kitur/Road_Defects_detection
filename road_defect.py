# -*- coding: utf-8 -*-
"""Road_Defect.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OfS3kPy_IoW6g8bsD98kCUEoAPOyLvhp
"""

ROOT_DIR ='/content/drive/MyDrive/sharon/road_defects/data'

!pip install ultralytics

import os
from ultralytics import YOLO

model = YOLO('yolov8m.yaml')
resutls = model.train(data=os.path.join(ROOT_DIR, 'data.yaml'), epochs=80)

"""# New Section"""

import shutil

# Compress the folder
shutil.make_archive('train_results', 'zip', 'runs/detect/train')

# Download the zip file
from google.colab import files
files.download('train_results.zip')

import os

test =['/content/drive/MyDrive/sharon/multi-label/'+y for y in os.listdir('/content/drive/MyDrive/sharon/multi-label') ]

test[:10]

import matplotlib.image as mpimg # Import the necessary library for image loading

import matplotlib.pyplot as plt

def show_42images(images):
  plt.figure(figsize=(50, 50))
  for i in range(42):
    results= model.predict(images[i])
    axs = plt.subplot(7,6, i+1)
    img= mpimg.imread(images[i])
    plt.imshow(img)
    plt.axis('off')
  plt.show()

show_42images(test)

from ultralytics import YOLO

# Load a model
model = YOLO("/content/drive/MyDrive/sharon/road_defects/best.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
# results = model(test, stream=True)  # return a generator of Results objects

# # Process results generator
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk

res =model.predict(test[3])
res[0].show()

model

print(model)

from google.colab import files
files.download('/content/runs/detect/train/weights/best.pt')

from ultralytics import YOLO

model=YOLO('/content/runs/detect/train2/weights/best_m.pt')

see=model.predict(test[17])
see[0].show()



see

# prompt: save the yolo model. use this path'/content/drive/MyDrive/sharon'
# Ensure the model has the checkpoint data before saving
if model.ckpt is not None:
  model.save('/content/drive/MyDrive/sharon/my_yolov8_model.pt')
else:
  print("Model checkpoint data is missing. Please train or load the model with checkpoint information.")

