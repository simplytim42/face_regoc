import os
import cv2
import numpy as np

from PIL import Image

names = []
paths = []

for user in os.listdir("dataset"):
    names.append(user)

for name in names:
    for image in os.listdir("dataset/{}".format(name)):
        path_string = os.path.join("dataset", name, image)
        paths.append(path_string)


faces = []
ids = []

for img_path in paths:
    #convert to greyscale
    image = Image.open(img_path).convert('L')

    #convert to numpy array
    imgNp = np.array(image, 'uint8')

    faces.append(imgNp)

    # grab the id of the user
    id = int(img_path.split('/')[2].split('_')[0])
    
    ids.append(id)


# convert to numpy array
ids = np.array(ids)

# initialise face recogniser, train the model and save
trainer = cv2.face.LBPHFaceRecognizer_create()
trainer.train(faces, ids)
trainer.write("training.yml")