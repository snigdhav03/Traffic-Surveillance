# import necessary packages
from flask import Flask, render_template, Response
from imutils.video import VideoStream
import numpy as np
from imutils.video import FPS
import imutils
from time import time
import cv2
from keras.models import load_model
from camera import Camera

app= Flask(__name__)


@app.route("/", methods=["GET"])
def homepage():
    return render_template("index.html")

@app.route("/approach", methods=["GET"])
def approach():
    return render_template("approach.html")

@app.route("/challenges", methods=["GET"])
def challenges():
    return render_template("challenges.html")

@app.route("/display", methods=["GET"])
def display():
    return render_template("display.html")

def gen(cam):
    while True:
        frame = cam.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 

@app.route("/video", methods=["GET"])
def video():
    return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)