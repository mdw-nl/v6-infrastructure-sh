# Build recipe for the ADMM logistic regression algorithm. Originally
# vendored from mdw-nl/20kChallengeVantage6 (my-fl-project/20kLogRegChallenge)
# as a setuptools package; restructured into this repo's single-module
# layout (see the other algorithms/*/Dockerfile), so a plain Python base
# image with its dependencies installed directly is enough - no setup.py.
#
# Build (from this repo's root), same convention as the other algorithms/:
#   docker build -t admm_logistic_regression:latest algorithms/admm_logistic_regression/

FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "vantage6-algorithm-tools==4.14.0" \
    pandas \
    numpy \
    scipy \
    scikit-learn

WORKDIR /app
COPY admm_logistic_regression.py .

ENV PKG_NAME=admm_logistic_regression
CMD ["python", "-c", "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"]
