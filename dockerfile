FROM python:3.11-slim

RUN apt-get update

RUN pip3 install requests

ENTRYPOINT ["tail", "-f", "/dev/null"]