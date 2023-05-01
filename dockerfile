FROM python:3.11-slim

RUN apt-get update

COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN python -m pip install -e .

ENTRYPOINT ["tail", "-f", "/dev/null"]