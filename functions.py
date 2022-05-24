import cv2

from pathlib import Path

def saveImage(img, user_name, user_id, img_id):
    Path('dataset/{}'.format(user_name)).mkdir(parents=True, exist_ok=True)
    cv2.imwrite("dataset/{}/{}_{}.jpg".format(user_name, user_id, img_id), img)
