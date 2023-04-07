FROM basicai/xtreme1-image-vect-visualization-cpu

RUN pip install umap-learn==0.5.3

RUN rm -rf /home/*

COPY server.py /home

WORKDIR /home

EXPOSE 5000

ENTRYPOINT python -u server.py --port=5000
