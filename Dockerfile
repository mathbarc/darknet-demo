FROM python:3.9.15-slim

WORKDIR /home
COPY api/app.py api/yolo.py PSE_detector.cfg PSE_detector.weights class.name ./

RUN apt update;DEBIAN_FRONTEND=noninteractive apt install libavcodec-dev libavformat-dev libavresample-dev libswscale-dev -y; apt clean --dry-run; apt autoclean; pip3 install opencv-python Flask flask-cors

ENV FLASK_APP="app.py"

ENTRYPOINT [ "flask", "run" ]