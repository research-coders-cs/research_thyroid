SHELL := /bin/bash

# 'test' or 'ci'
TARGET ?= test
ci:
	TARGET=ci make test

test:
	@echo "(make sure \`pipenv install\`)"
	rm -rf result && mkdir result
	pipenv run python3 test.py

