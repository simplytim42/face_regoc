import cv2
import os

# initialise the classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# initialise recogniser and read the model
recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read("training.yml")

# grab the names of all people in dataset
names = []
for user in os.listdir("dataset"):
    names.append(user)

vc = cv2.VideoCapture(0)

while True:
    _, img = vc.read()

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #detect faces within the image
    faces = face_cascade.detectMultiScale(grey_img, scaleFactor=1.2, minNeighbors=5, minSize=(50,50))

    # draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(
            img, 
            (x, y), # bottom left of the rectangle
            (x+w, y+h), # top right of the rectangle
            (255, 0, 0), # border colour
            2) # border width
        
        # make a prediction as to who this is
        id, confidence = recogniser.predict(grey_img[y:y+h, x:x+w])
        
        # place name around the face on the image
        if id:
            cv2.putText(
                img,
                names[id-1] + ". Confidence: " + str(int(confidence)) + "%", # the name of the user
                (x, y-4), # coords of the text
                cv2.FONT_HERSHEY_SIMPLEX, # ?
                0.8, # ?
                (0, 255, 0), # ?
                1, # ?
                cv2.LINE_AA) # ?
        else:
            cv2.putText(
                img,
                'Unknown',
                (x, y-4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                1,
                cv2.LINE_AA) 

    cv2.imshow("Recognise", img)

    # wait for user to click the q key to close image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()