# Sanitinet

## Overview

Sanitinet is an API server that detects hate speech in text and NSFW content in images, tailored to handle hinglish and multiple regional indic languages for jeets.

## setup

- install locally

```
make install
```

- build using docker

```
make docker-build
```

## run

- run locally

```
make run
```

- using docker

```
make docker-run
```

## Usage

- classify image

```
curl --request POST \
  --url /v1/image/classify \
  --header 'Content-Type: multipart/form-data' \
  --form input=@image.png
```

```json
{
  "nsfw": false,
  "raw": [
    {
      "label": "drawings",
      "score": 0.9997829794883728
    },
    {
      "label": "neutral",
      "score": 0.0000691982641001232
    },
    {
      "label": "hentai",
      "score": 0.000057825262047117576
    },
    {
      "label": "porn",
      "score": 0.0000463492760900408
    },
    {
      "label": "sexy",
      "score": 0.000043643511162372306
    }
  ]
}
```

- classify text

```
curl --request POST \
  --url /v1/chat/classify \
  --header 'Content-Type: application/json' \
  --data '{
  "input_text":"hi how are you"
}'
```

```json
{
  "is_hate_speech": false,
  "input": "hi how are you",
  "raw": {
    "label": "LABEL_0",
    "score": 0.9889931082725525
  }
}
```
