FROM python:3.8-slim-buster

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

WORKDIR /app
COPY flask_captioning.py /app

# COPY requirements.txt requirements.txt

# Install pycocotools
RUN apt-get install -y gcc
#RUN pip install Cython
#RUN pip install numpy
#RUN pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

# Install other requirements
#RUN pip install -r requirements.txt
#RUN python -m nltk.downloader punkt && python -m nltk.downloader stopwords

EXPOSE 80

CMD ["python", "hello.py"]
