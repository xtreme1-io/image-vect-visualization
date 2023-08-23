FROM python:3.8-bullseye

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN wget -P $HOME/.cache/torch/hub/checkpoints/ https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth mobilenet_v3_small-047dcff4.pth

COPY . ./

WORKDIR /app/src
EXPOSE 5000
CMD [ "python", "-u", "server.py", "--port=5000" ] 
