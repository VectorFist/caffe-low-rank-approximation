from one_shot_compression import *

proto_input = "examples/cifar10/cifar10_quick_train_test.prototxt"
weights_input = "examples/cifar10/cifar10_quick_iter_5000.caffemodel.h5"
proto_output = "cifar10_quick_lra_train_text.prototxt"
weights_output = "cifar10_quick_lra.caffemodel.h5"

if __name__ == "__main__":
    one_shot_compression(proto_input, weights_input, proto_output, weights_output, True)
