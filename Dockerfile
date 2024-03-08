FROM python:3.7.5

WORKDIR /app
COPY . .

ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

RUN pip install -r microminer_eval/algorithm/requirements.txt --pre

RUN pip install Cython==0.29.36 \
    && pip install scikit-learn==0.24.2 --no-build-isolation

RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;

RUN path="$PWD" \
    && JAVA_TOOL_OPTIONS=-Dfile.encoding=UTF8 \
    && apt install ant -y \
    && git clone https://github.com/mimno/Mallet.git ~/.mallet \
    && cd ~/.mallet \
    && ant
