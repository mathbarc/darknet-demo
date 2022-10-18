FROM python:3.9.15-slim

WORKDIR /home
COPY api/app.py .
COPY api/yolo.py .
COPY PSE_detector.cfg .
COPY PSE_detector.weights .
COPY class.name .

RUN apt update;DEBIAN_FRONTEND=noninteractive apt install libavcodec-dev libavformat-dev libavresample-dev libswscale-dev libeigen3-dev -y; apt clean --dry-run; apt autoclean; pip3 install opencv-python opencv-contrib-python Flask flask-cors

ENV FLASK_APP="app.py"

ENTRYPOINT [ "flask", "run" ]