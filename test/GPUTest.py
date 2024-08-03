# check both Pytorch and Tensorflow
import torch
import tensorflow as tf

print(torch.__version__)
print(torch.version.cuda)  # CUDA version
print(torch.backends.cudnn.version())  # cuDNN version
print(tf.sysconfig.get_build_info()['cuda_version'])
print(tf.sysconfig.get_build_info()["cudnn_version"])
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

print(tf.config.list_physical_devices('GPU'))
print(tf.config.list_physical_devices('CPU'))
print(tf.test.gpu_device_name())
# '/device:GPU:0'
print(tf.config.list_logical_devices())

