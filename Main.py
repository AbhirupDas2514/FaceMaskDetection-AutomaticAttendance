def face_recog():
    import cv2
    import numpy as np
    import face_recognition
    import os
    from datetime import datetime

    path = 'training_images/employees'
    images = []
    classNames = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)

    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    def markAttendance(name):
        with open('AttendanceSheet.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = []

            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                datee=now.strftime("%m/%d/%y")
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{datee},{dtString}')
                print(f"\n{name} may enter")

    encodeListKnown = findEncodings(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    count = 0
    while True:

        success, img = cap.read()

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0))
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        cv2.imshow('Webcam', img)
        cv2.waitKey(2)
        if len(facesCurFrame) > 0:
            count+=1
        if count > 3:
            break
    cv2.destroyAllWindows()
#FACE_MASK_CODE

import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haar_data.xml')
mouth_cascade = cv2.CascadeClassifier('mouth.xml')
# User message
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
weared_mask_font_color = (255, 255, 255)
not_weared_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
weared_mask = "Thank You for wearing MASK,Enter"
not_weared_mask = "Please wear MASK before entering"
textt=""

# Read video
cap = cv2.VideoCapture(0)
count=0
while 1:
    # Get individual frame
    ret, img = cap.read()
    img = cv2.flip(img,1)

    # Convert Image into gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    # detect face
    faces = face_cascade.detectMultiScale(gray)

    if(len(faces) == 0):
        cv2.putText(img, "No face found...", org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)

    else:
        # Draw rectangle on face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

            # Detect lips counters
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)
        # Face detected but Lips not detected which means person is wearing mas
        # k
        if(len(mouth_rects) == 0):
            cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
            textt="Thank You for wearing MASK,Enter"
            count+=1
            if count>15:
                break
        else:
            for (mx, my, mw, mh) in mouth_rects:

                if(y < my < y + h):
                    cv2.putText(img, not_weared_mask, org, font, font_scale, not_weared_mask_font_color, thickness, cv2.LINE_AA)

                    break

    # Show frame with results
    cv2.imshow('Mask Detection', img)
    k = cv2.waitKey(30)
    if k == 27:
        break

# Release video
cap.release()
cv2.destroyAllWindows()
if (textt=="Thank You for wearing MASK,Enter"):
    face_recog()
