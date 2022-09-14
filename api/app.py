import yolo
from flask import Flask, request, jsonify
import numpy
import base64
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

labelsFile = open("class.name", "r")
labelsList = [label.strip() for label in labelsFile.readlines()]
model = yolo.Yolo("yolov4-tiny.cfg", "backup/yolov4-tiny_final.weights",labelsList, (416, 416), 0.3)


def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = numpy.fromstring(base64.b64decode(encoded_data), numpy.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

def writeb64(img):
    _, encoded = cv2.imencode(".jpeg", img)
    base64String = base64.b64encode(encoded)
    return base64String.decode('utf-8')

@app.route("/detect", methods=["POST"])
def detect():
    
    imageBase64 = request.form.get("image")
    img = readb64(imageBase64)    
    result = model.inference(img)

    for det in result:
        cv2.rectangle(img, det["roi"], (0,255,0))
    cv2.imwrite("test.png", img)

    output = {}
    output["image"] = "data:image/jpeg;base64,"+str(writeb64(img))
    output["detections"] = result

    return jsonify(output)


if __name__=="__main__":
    app.run()