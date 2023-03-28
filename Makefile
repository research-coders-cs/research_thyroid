SHELL := /bin/bash

ci:
	pipenv install
	##make test-legacy  # dev/debug
	make test

DL_ASSETS := https://github.com/research-coders-cs/research_thyroid/releases/download/assets-0.1

#

Dataset_train_test_val:
	curl -O -L $(DL_ASSETS)/$@.rar
	unrar x $@.rar
net_debug.pth:
	curl -O -L $(DL_ASSETS)/$@
test-legacy: Dataset_train_test_val net_debug.pth
	rm -rf log_legacy.txt result_legacy && mkdir result_legacy
	pipenv run python3 main_legacy.py 2>&1 | tee log_legacy.txt
	zip -r result_legacy.zip result_legacy > /dev/null

#

WSDAN_densenet_224_16_lr-1e5_n1-remove_220828-0837_85.714.ckpt:
	curl -O -L $(DL_ASSETS)/$@
WSDAN_doppler_densenet_224_16_lr-1e5_n5_220905-1309_78.571.ckpt:
	curl -O -L $(DL_ASSETS)/$@
densenet_224_8_lr-1e5_n4_95.968.ckpt:
	curl -O -L $(DL_ASSETS)/$@
Siriraj_sample_doppler_comp:
	curl -O -L $(DL_ASSETS)/$@.zip
	unzip $@.zip
Siriraj_sample_doppler_20:
	curl -O -L $(DL_ASSETS)/$@.zip
	unzip $@.zip
test: Dataset_train_test_val \
	WSDAN_densenet_224_16_lr-1e5_n1-remove_220828-0837_85.714.ckpt \
	WSDAN_doppler_densenet_224_16_lr-1e5_n5_220905-1309_78.571.ckpt \
	densenet_224_8_lr-1e5_n4_95.968.ckpt \
	Siriraj_sample_doppler_comp \
	Siriraj_sample_doppler_20
	rm -rf log.txt output && mkdir output
	time pipenv run python3 main.py 2>&1 | tee log.txt
	zip -r output.zip output > /dev/null

