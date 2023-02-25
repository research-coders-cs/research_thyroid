SHELL := /bin/bash

ci:
	pipenv install
	##make test-legacy
	make test

DL_ASSETS := https://github.com/research-coders-cs/research_thyroid/releases/download/assets-0.1
DATASET_ZIP := Dataset_train_test_val.rar
Dataset_train_test_val:
	curl -O -L $(DL_ASSETS)/$(DATASET_ZIP)
	unrar x $(DATASET_ZIP)


net_debug.pth:
	curl -O -L $(DL_ASSETS)/$@
test-legacy: Dataset_train_test_val net_debug.pth
	rm -rf log_legacy.txt result_legacy && mkdir result_legacy
	pipenv run python3 test_legacy.py 2>&1 | tee log_legacy.txt
	zip -r result_legacy.zip result_legacy > /dev/null


WSDAN_densenet_224_16_lr-1e5_n1-remove_220828-0837_85.714.ckpt:
	curl -O -L $(DL_ASSETS)/$@
WSDAN_doppler_densenet_224_16_lr-1e5_n5_220905-1309_78.571.ckpt:
	curl -O -L $(DL_ASSETS)/$@
test: Dataset_train_test_val WSDAN_densenet_224_16_lr-1e5_n1-remove_220828-0837_85.714.ckpt WSDAN_doppler_densenet_224_16_lr-1e5_n5_220905-1309_78.571.ckpt
	rm -rf log.txt result && mkdir result
	pipenv run python3 test.py | tee log.txt
	zip -r result.zip result > /dev/null

