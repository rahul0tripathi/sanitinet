FROM python:3-slim as compiler
ENV PYTHONUNBUFFERED 1

WORKDIR /app/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN apt-get update
RUN apt-get -y install gcc
RUN apt-get -y install git


COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY ./install.sh /app/install.sh
RUN sh install.sh


FROM python:3-slim as runner
WORKDIR /app/
COPY --from=compiler /opt/venv /opt/venv
COPY --from=compiler /app/IndicTransTokenizer/IndicTransTokenizer /app/IndicTransTokenizer
ENV PATH="/opt/venv/bin:$PATH"
RUN python3 -c "import nltk ; nltk.download('punkt')"
COPY ./snapshot /app/snapshot
COPY ./*.py /app/
CMD ["python3", "main.py"]
