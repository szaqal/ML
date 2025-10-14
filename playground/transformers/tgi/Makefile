#https://github.com/huggingface/text-generation-inference/pkgs/container/text-generation-inference
start-tgi:
	docker run \
	    --shm-size 1g \
	    -p 8080:80 \
	    -v ~/.cache/huggingface:/data \
	    ghcr.io/huggingface/text-generation-inference:3.3.6-trtllm \
	    --model-id HuggingFaceTB/SmolLM2-360M-Instruct
