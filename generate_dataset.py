import cv2
import copy

from functions import saveImage

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

vc = cv2.VideoCapture(0)

print("Enter the User ID and Name:")
user_id = input("User's ID: ")
user_name = input("User's Name: ")

count = 1    

while True:
    _, img = vc.read()

    # make copy of original image for saving purposes
    original_img = copy.copy(img)

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #detect faces within the image
    faces = face_cascade.detectMultiScale(grey_img, scaleFactor=1.2, minNeighbors=5, minSize=(50,50))

    # draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(
            img, 
            (x, y), # bottom left of the rectangle
            (x+w, y+h), # top right of the rectangle
            (0, 255, 0), # border colour
            2) # border width

    cv2.imshow("Identified Faces", img)

    #grab the key that was clicked
    key = cv2.waitKey(1) & 0xFF

    # save image or stop the loop
    if key == ord('s'):
        saveImage(original_img, user_name, user_id, count)
        count += 1
    elif key == ord('q'):
        break

# clean up
vc.release()
cv2.destroyAllWindows()