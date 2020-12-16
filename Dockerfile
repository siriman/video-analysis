FROM siriman/spaimage:version1
MAINTAINER "Siriman" "sk@sensor6ty.com"
RUN pip install flask
#We copy just the requirements.txt first to leverage Docker cache
COPY . /home/yolov4-deepsort/
WORKDIR /home/yolov4-deepsort
#RUN pip3 install --upgrade pip
#RUN pip3 install -r requirements.txt
#COPY . /app
ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]
