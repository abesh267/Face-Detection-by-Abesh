import cv2

capture = cv2.VideoCapture(0)
cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

while True:
    ret, window = capture.read()
    window = cv2.cvtColor(window,0)
    detections = cascade_classifier.detectMultiScale(window)
    if len(detections) > 0:
        (x,y,w,h) = detections[0]
        window = cv2.rectangle(window,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('Face Window',window)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()


'''
Used Haarcascades Classifier
'''
