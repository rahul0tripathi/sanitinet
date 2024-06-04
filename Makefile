.PHONY: docker-build
docker-build:
	@python load_models.py
	@docker build . -t sanitinet:latest


.PHONY: docker-run
docker-run:
	@docker run -p 8000:8000 sanitinet:latest

.PHONY: run
run:
	@python main.py

.PHONY: install 
install:
	@pip install -r requirements.txt
	@sh install.sh