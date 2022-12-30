FROM basicai/xtreme1-image-vect-visualization-cpu

RUN rm -rf /home

COPY server.py /home

WORKDIR /home

EXPOSE 5000

ENTRYPOINT python -u server.py --port=5000
