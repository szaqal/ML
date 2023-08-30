
run-cpu: export CUDA_VISIBLE_DEVICES=
run-cpu:
	./mnist_tf.py

run-gpu: export TF_MIN_GPU_MULTIPROCESSOR_COUNT=1
run-gpu: export CUDA_VISIBLE_DEVICES=0,1
run-gpu:
	./mnist_tf.py
