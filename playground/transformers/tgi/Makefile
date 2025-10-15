#https://github.com/huggingface/text-generation-inference/pkgs/container/text-generation-inference
start-tgi:
	docker run  \
            --gpus all \
	    -e DEVICE_MAP='{"": "cpu"}'  \
	    -e CUDA_VISIBLE_DEVICES= \
	    --shm-size 1g \
	    --env CUDA_VISIBLE_DEVICES= \
	    --env DEVICE_MAP="{"":"cpu"}" \
	    -p 8080:80 \
	    -v ~/.cache/huggingface:/data \
	    ghcr.io/huggingface/text-generation-inference:3.3.6-trtllm \
	    --model-id HuggingFaceTB/SmolLM2-360M-Instruct
