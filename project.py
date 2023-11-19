from json import load
import cv2 as cv
import numpy as np
import pickle
import os

vid = cv.VideoCapture('./videos/videoplayback.mp4')
pickle_in = open(
    '/home/aditya/projects/Data-Science-Project/Data-Science-Project-main/Traffic Signal Rec/modelcar1.p', 'rb')
model = pickle.load(pickle_in)
# os.chdir('/Users/rajsinghbani/Documents/Traffic Signal Rec/0_s')

def getBox(frame, cascade, upperIgnore=0.1, bilateralParams=(9, 75, 75), scale=5, minNeighbors=4):
    c = 1
    (H, W) = frame.shape[:2]
    upperIgnore = int(upperIgnore*H)
    test = frame[upperIgnore:, :]
    test = cv.cvtColor(test, cv.COLOR_BGR2GRAY)
    (a, b, c) = bilateralParams
    test = cv.bilateralFilter(test, a, b, c)
    faces, rejects, weights=cascade.detectMultiScale3(test, scale, minNeighbors, outputRejectLevels=True, minSize=(30, 30), maxSize=(300,300))
    return faces, weights

def make_box(frame):
    (H, W) = frame.shape[:2]
    upperIgnore = int(0.1*H)
    faces, weights = getBox(frame, cascade, 0.1)
    l = []
    c = 1
    #print(faces)
    notNullBoxes = []
    for (x, y, w, h) in faces:
        try:
            cropped = frame[upperIgnore+y:upperIgnore+y+h, x:x+w]
            cropped = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
            cropped = cv.resize(cropped, (32, 32), interpolation=cv.INTER_AREA)
            l.append(cropped)
            notNullBoxes.append((x, y, w, h))
        except:
            pass
        c += 1
    l = np.array(l)
    l = l/255
    l = l.reshape((-1, 32, 32, 1))
    if len(l) != 0:
        pred = np.argmax(model.predict(l), axis=1)
        print(pred)

    counter = 0
    boxes = []
    # for (x, y, w, h) in notNullBoxes:
    #     if
    for (x, y, w, h) in notNullBoxes:
        if pred[counter] == 1:
            frame = cv.rectangle(frame, (x, upperIgnore+y),
                                 (x+w,  upperIgnore+y+h), (0, 255, 0), 3)
            # test = frame[upperIgnore+y:upperIgnore+y+h, x:x+w]
            # cv.imwrite('/Users/rajsinghbani/Documents/Traffic Signal Rec/0_s/test'+f'{counter}'+f'{d}'+'.png', test)
            # test = cv.cvtColor(test, cv.COLOR_RGB2GRAY)
            # test = cv.resize(test, (32, 32), interpolation=cv.INTER_AREA)
            # k = []
            # k.append(test)
            # k = np.array(k)
            # k = k/255
            # k = k.reshape((-1,32,32,1))
            # p = np.argmax(model.predict(l),axis=1)
            # if p[0] == 1:
            #     print('yes')
            # cv.imshow('n', test)

        counter += 1
    return frame


cascade = cv.CascadeClassifier(
    '/home/aditya/projects/Data-Science-Project/Data-Science-Project-main/Traffic Signal Rec/cascade.xml')
d = 1
while 1:
    ok, frame = vid.read()
    if not ok:
        break
    frame = make_box(frame)
    cv.imshow('Video', frame)
    if cv.waitKey(30) & 0xff == ord('q'):
        break
    d += 1

cv.destroyAllWindows()