#!/usr/bin/python
import cv2
import time
from imutils.video import VideoStream
import numpy as np
from imutils.video import FPS
import imutils
from keras.models import load_model

#!/usr/bin/python
import cv2
import time

class Camera():
    # Constructor...
    def __init__(self,arg):
        w = 640			# Frame width...
        h = 480			# Frame hight...
        fps = 20.0                    # Frames per second...
        resolution = (w, h)         	# Frame size/resolution...
        self.video_name=arg

        # initialize the list of class labels MobileNet SSD was trained to detect
        # generate a set of bounding box colors for each class
        CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        #CLASSES = ['motorbike', 'person']
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(
            'MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

        print('Loading helmet model...')
        loaded_model = load_model('new_helmet_model.h5')
        loaded_model.compile(loss='binary_crossentropy',
                            optimizer='rmsprop', metrics=['accuracy'])

        # initialize the video stream,
        print("[INFO] starting video stream...")

        # Loading the video file
        self.cap = cv2.VideoCapture(self.video_name)
        print("Camera warming up ...")
        time.sleep(1)

        # Starting the FPS calculation
        fps = FPS().start()
        # self.ret, self.frame = self.cap.read()

        # loop over the frames from the video stream
        # i = True

        # grab the frame from the threaded video stream and resize it
        # to have a maxm width and height of 600 pixels
        while True: 
            try:
                self.ret, self.frame = self.cap.read()

                # resizing the images
                self.frame = imutils.resize(self.frame, width=600, height=600)

                # grab the frame dimensions and convert it to a blob
                (h, w) = self.frame.shape[:2]

                # Resizing to a fixed 300x300 pixels and normalizing it.
                # Creating the blob from image to give input to the Caffe Model
                blob = cv2.dnn.blobFromImage(cv2.resize(
                    self.frame, (300, 300)), 0.007843, (300, 300), 127.5)

                # pass the blob through the network and obtain the detections and predictions
                net.setInput(blob)

                detections = net.forward()  # getting the detections from the network

                persons = []
                person_roi = []
                motorbi = []

                # loop over the detections
                for i in np.arange(0, detections.shape[2]):
                    # extract the confidence associated with the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the confidence
                    # is greater than minimum confidence
                    if confidence > 0.5:

                        # extract index of class label from the detections
                        idx = int(detections[0, 0, i, 1])

                        if idx == 15:
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            # roi = box[startX:endX, startY:endY/4]
                            # person_roi.append(roi)
                            persons.append((startX, startY, endX, endY))

                        if idx == 14:
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            motorbi.append((startX, startY, endX, endY))

                xsdiff = 0
                xediff = 0
                ysdiff = 0
                yediff = 0
                p = ()

                for i in motorbi:
                    mi = float("Inf")
                    for j in range(len(persons)):
                        xsdiff = abs(i[0] - persons[j][0])
                        xediff = abs(i[2] - persons[j][2])
                        ysdiff = abs(i[1] - persons[j][1])
                        yediff = abs(i[3] - persons[j][3])

                        if (xsdiff+xediff+ysdiff+yediff) < mi:
                            mi = xsdiff+xediff+ysdiff+yediff
                            p = persons[j]
                            # r = person_roi[j]
                        
                    if len(p) != 0:

                        # display the prediction
                        label = "{}".format(CLASSES[14])
                        print("[INFO] {}".format(label))
                        cv2.rectangle(
                            self.frame, (i[0], i[1]), (i[2], i[3]), COLORS[14], 2)
                        y = i[1] - 15 if i[1] - 15 > 15 else i[1] + 15
                        cv2.putText(
                            self.frame, label, (i[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[14], 2)
                        label = "{}".format(CLASSES[15])
                        print("[INFO] {}".format(label))

                        cv2.rectangle(
                            self.frame, (p[0], p[1]), (p[2], p[3]), COLORS[15], 2)
                        y = p[1] - 15 if p[1] - 15 > 15 else p[1] + 15

                        roi = self.frame[p[1]:p[1]+(p[3]-p[1])//4, p[0]:p[2]]
                        print(roi)
                        if len(roi) != 0:
                            img_array = cv2.resize(roi, (50, 50))
                            gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                            img = np.array(gray_img).reshape(1, 50, 50, 1)
                            img = img/255.0
                            prediction = loaded_model.predict_proba([img])
                            cv2.rectangle(
                                self.frame, (p[0], p[1]), (p[0]+(p[2]-p[0]), p[1]+(p[3]-p[1])//4), COLORS[0], 2)
                            cv2.putText(self.frame, str(round(
                                prediction[0][0], 2)), (p[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)
                
                        
            except:
                pass

            cv2.imshow('Frame', self.frame)  # Displaying the frame
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'): # if 'q' key is pressed, break from the loop
                break
            # update the FPS counter
            fps.update()
        
        cv2.destroyAllWindows()
        #cap.release()


    # Frame generation for Browser streaming wiht Flask...

    def get_frame(self):
        self.frames = open("stream.jpg", 'wb+')
        s, img = self.cap.read()      
        if s:  # frame captures without errors...
            cv2.imwrite("stream.jpg", img)  # Save image...
        return self.frames.read()

    def diffImg(self, tprev, tc, tnex):
        # Generate the 'difference' from the 3 captured images...
        Im1 = cv2.absdiff(tnex, tc)
        Im2 = cv2.absdiff(tc, tprev)
        return cv2.bitwise_and(Im1, Im2)

    def captureVideo(self):
        # Read in a new frame...
        self.ret, self.frame = self.cap.read()
        # Image manipulations come here...
        # This line displays the image resulting from calculating the difference between
        # consecutive images...
        diffe = self.diffImg(
            self.prev_frame, self.current_frame, self.next_frame)
        cv2.imshow(self.winName, diffe)

        # Put images in the right order...
        self.prev_frame = self.current_frame
        self.current_frame = self.next_frame
        self.next_frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        return()

    def saveVideo(self):
        # Write the frame...
        self.out.write(self.frame)
        return()

    


def main():
    # Create a camera instance...
    while(True):
        # Display the resulting frames...
              # Save video to file 'output.avi'...
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return()


if __name__ == '__main__':
   main()
