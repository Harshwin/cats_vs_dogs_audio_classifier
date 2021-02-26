FROM python:3.6.5-slim
WORKDIR /audio_classifier
COPY . /audio_classifier
LABEL maintainer="harshwin1693@gmail.com"

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y \
&& apt-get -y install apt-utils gcc libpq-dev libsndfile-dev
RUN pip install --upgrade pip && pip install -r requirements.txt
EXPOSE 8888

ENTRYPOINT [ "python" ]
CMD [ "pipeline.py", "-T", "-P"]