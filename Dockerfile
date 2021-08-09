FROM python:3.8-slim-buster

WORKDIR /SECOM_Flask
ADD . /SECOM_Flask

RUN pip3 install -r requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/SECOM_Flask"
CMD ["python", "API/app.py"]