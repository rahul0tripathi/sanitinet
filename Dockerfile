FROM python:3.9-slim as compiler
ENV PYTHONUNBUFFERED 1

WORKDIR /app/

RUN python -m venv /opt/venv
# Enable venv
ENV PATH="/opt/venv/bin:$PATH"
RUN apt-get update
RUN apt-get -y install gcc
RUN apt-get -y install git


COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY ./install.sh /app/install.sh
RUN sh install.sh
RUN echo $PWD
RUN echo $(ls -al)

FROM python:3.9-slim as runner
WORKDIR /app/
COPY --from=compiler /opt/venv /opt/venv
COPY --from=compiler /app/IndicTransTokenizer /app/IndicTransTokenizer
# Enable venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./main.py /app/
CMD ["python3", "main.py"]
