import cv2 #importing opencv

cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')  #this is an XML lib to select the object you wanna detect

cap = cv2.VideoCapture(0)   #to make it a video frame

while True:
    ret, frame = cap.read()     # Capture frame-by-frame
    gray = cv2.cvtColor(frame, 0)  # Our operations on the frame come here
    detections = cascade_classifier.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    if(len(detections) > 0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # for (x,y,w,h) in detections:
    # 	frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
