import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

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
            (0, 255, 0), # border colour
            2) # border width

    cv2.imshow("Identified Faces", img)

    # wait for user to click the q key to close image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()