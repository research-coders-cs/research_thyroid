SHELL := /bin/bash

ci:
	pipenv install
	make test

test:
	@echo "(first time, make sure \`pipenv install\`)"
	rm -rf result && mkdir result
	pipenv run python3 test.py

