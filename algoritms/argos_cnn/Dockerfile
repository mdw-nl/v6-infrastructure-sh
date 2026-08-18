FROM python:3.10-slim

RUN pip install --no-cache-dir \
    "vantage6-algorithm-tools==4.14.0" \
    pandas \
    numpy \
    nibabel && \
    pip install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY model.py .
COPY argos_cnn.py .

ENV PKG_NAME=argos_cnn
CMD ["python", "-c", "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"]
