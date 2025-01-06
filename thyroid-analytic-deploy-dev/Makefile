
help:
	@echo ---- To visit the reference deploy on remote servers:
	@echo ssh tha
	@echo ssh thw
	@echo ---- To test triggering local deploy:
	@echo make test

WAIT_SECS := 10
INFERENCE_MODEL := densenet121_250_8_lr-1e5_n4_79.592
INFERENCE_IMAGE := benign_nodule1_0001-0100_c0008_1_p0016.png
INFERENCE_DOPPLER := true
#INFERENCE_DOPPLER := false# without GPU getting, expect -- RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
ENDPOINT := https://localhost:9999/inference?=api-key=ThyroidInference
test:
	# open Docker Desktop (with the container auto-run likely)
	#---- in a separate shell
	# (docker compose stop)
	# (docker ps)
	# docker compose up  # for minitoring log; `--build` likely not necessary
	#----
	docker compose restart && sleep $(WAIT_SECS) && \
		curl -k -X POST -H "Content-Type: application/json" -d '{"model_path": "$(INFERENCE_MODEL)", "isDoppler": "$(INFERENCE_DOPPLER)", "image_path": "$(INFERENCE_IMAGE)"}' $(ENDPOINT) && \
		say ok || say error
