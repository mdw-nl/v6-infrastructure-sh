FROM python:3.10-slim

RUN pip install --no-cache-dir \
    "vantage6-algorithm-tools==4.14.0" \
    pandas \
    numpy \
    scipy

WORKDIR /app
COPY linear_regression.py .

ENV PKG_NAME=linear_regression
CMD ["python", "-c", "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"]
