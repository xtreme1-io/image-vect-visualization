FROM ubuntu:3.8-bullseye

RUN pip install umap-learn==0.5.3

WORKDIR /app
COPY . ./

WORKDIR /app/src
EXPOSE 5000
ENTRYPOINT python -u server.py --port=5000
