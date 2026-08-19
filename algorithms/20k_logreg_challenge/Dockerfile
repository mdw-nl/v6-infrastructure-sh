# Build recipe for the ADMM logistic regression algorithm. The algorithm
# source (setup.py, logreg_challenge_20k/, README.md) is vendored directly
# into this folder - copied from mdw-nl/20kChallengeVantage6
# (my-fl-project/20kLogRegChallenge) - so this repo alone is enough to build
# the image; no separate clone of that repo is needed. Since it's a copy, not
# a live reference, re-sync it manually if the upstream algorithm changes.
#
# The upstream Dockerfile there is FROM harbor2.vantage6.ai/infrastructure/
# algorithm-base, which no longer resolves (Harbor was retired in favor of
# GHCR - see this repo's own README). vantage6-algorithm-tools is a normal
# PyPI package and is already an install_requires of the algorithm's own
# setup.py, so a plain Python base image is enough.
#
# Build (from this repo's root), same convention as the other algorithms/:
#   docker build -t 20klogregchallenge:latest algorithms/20k_logreg_challenge/

FROM python:3.12-slim

COPY . /app
RUN pip install --no-cache-dir /app

# wrap_algorithm() reads the module to run from the PKG_NAME env var at
# runtime (os.environ.get("PKG_NAME")) - it does NOT take the module name as
# a function argument (its real signature is wrap_algorithm(log_traceback:
# bool = True)). This ENV line is therefore required, not decorative.
ENV PKG_NAME=logreg_challenge_20k

CMD ["python", "-c", "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"]
