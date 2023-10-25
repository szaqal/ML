export BATCH_SIZE=100
export TF_MIN_GPU_MULTIPROCESSOR_COUNT=1
export EPOCHS=100
export BIAS =zeros
export LEARNING_RATE=0.01

#
# mean_squared_error
# categorical_crossentropy
#
export LOSS_FUNCTION=categorical_crossentropy

#
# Kernel initializer
# - glorot_uniform
# - random_uniform
# 
export KERNEL_INITIALIZER=glorot_uniform


#
# - sgd
# - adam
#
export OPTIMIZER=sgd

run-cpu: export CUDA_VISIBLE_DEVICES=
run-cpu:
	./mnist_tf.py

run-gpu: export CUDA_VISIBLE_DEVICES=0,1
run-gpu:
	./mnist_tf.py
