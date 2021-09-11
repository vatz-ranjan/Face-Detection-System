'''
*********************************************

--- WELCOME TO THIS FACE DETECTION SYSTEM ---

*********************************************
'''

'''
*****************************************************************

This program contains 4 functions and a main function

First Function : img_recog(face_cascade)
Detect the face in the image captured through web camera

Second Function : img_recog_path(face_cascade, img_path)
Detect the face (if present) in image provided by user

Third Function : video_recog(face_cascade)
Detect the face in the video captured through web camera

Fourth Function : video_recog_path(face_cascade, vid_path)
Detect the face (if present) in video provided by user

*****************************************************************
'''


import numpy as np
import cv2
import os

# This function is to detect the face in the image captured through web camera
def img_recog(face_cascade):
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret_val, img = video.read()

    if (ret_val):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05)
        print("Found {} faces which are @ {}".format(len(faces), faces))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Face", img)
        cv2.waitKey(0)
        video.release()
        cv2.destroyAllWindows()

    else: print("--- Image not found ---")
    return

# This function is to detect the face (if present) in image provided by user
def img_recog_path(face_cascade, img_path):

    try:
        img = cv2.imread(img_path, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05)
        print("Found {} faces which are :- {}".format(len(faces), faces))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except:
        print("\n-- Unable to open the image --\n")

    return

# This function is to detect the face (if present) in the video captured through web camera
def video_recog(face_cascade):
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while video.isOpened():
        check, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3)
        print("Found {} faces which are :- {}".format(len(faces), faces))

        for (a, y, w, h) in faces:
            cv2.rectangle(frame, (a, y), (a + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    return

# This function is to detect the face (if present) in video provided by user
def video_recog_path(face_cascade, vid_path):
    video = cv2.VideoCapture(vid_path)

    if (video.isOpened() == False):
        print("\n-- Unable to open the video --\n")
        return

    while video.isOpened():
        check, frame = video.read()
        if (check == True):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3)
            print("Found {} faces which are :- {}".format(len(faces), faces))

            for (a, y, w, h) in faces:
                cv2.rectangle(frame, (a, y), (a + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Image", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else: break

    video.release()
    cv2.destroyAllWindows()
    return

# Main Function
if __name__ == '__main__':

    intro = "--- WELCOME TO THIS FACE DETECTION SYSTEM ---"

    print("*"*len(intro))
    print("\n{}\n".format(intro))
    print("*" * len(intro))
    print("\n\n")

    cas_path = "C:/Users/RK/Desktop/Coding/PYC Python/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cas_path)

    again = True

    while again:

        img_or_vid = int(input("\n-- Image or Video (1. Image / 2. Video) : "))
        path = input("-- Enter path (0 for WebCam) : ")

        try:
            path = int(path)
            if (img_or_vid == 1): img_recog(face_cascade)
            elif (img_or_vid == 2):video_recog(face_cascade)

        except:
            path = r'{}'.format(path)
            path = os.path.join(path)
            if (img_or_vid == 1): img_recog_path(face_cascade, path)
            elif (img_or_vid == 2): video_recog_path(face_cascade, path)

        again_stat = """--- WANT TO DO MORE ---
        --- 1 for continue ---
        ---  2 to stop ---
        --- Enter your choice : """

        print("*"*len(intro))
        try:
            again_in = int(input("{}".format(again_stat)))
        except:
            print()
        print("*" * len(intro))

        if (again_in == 1): again = True
        else: again = False

    at_last = "--- THANKS, Hope you enjoyed this ---"

    print("*" * len(intro))
    print("\n{}\n".format(at_last))
    print("*" * len(intro))
    print("\n\n")



