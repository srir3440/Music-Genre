FROM tensorflow/tensorflow:2.9.1
WORKDIR /app
ADD . /app
RUN apt-get update
RUN apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
RUN pip install -r requirements.txt
CMD ["uwsgi","myapp.ini"]