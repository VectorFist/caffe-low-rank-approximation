import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from tensor_unfolder import unfold
from VBMF import VBMF
from tucker_decomposition import tucker_decomposition


def one_shot_compression(network_in, weights_in, network_out, weights_out, first_layer=False):
    layers_ranks = find_ranks(network_in, weights_in, first_layer)
    print(layers_ranks)
    proto_decomposition(network_in, network_out, layers_ranks)
    weights_decomposition(network_in, weights_in, network_out, weights_out, layers_ranks)


def find_ranks(network_in, weights_in, first_layer=False):
    network_caffe = caffe.Net(network_in, weights_in, caffe_pb2.TEST)
    layers_ranks = dict()
    last_layer = None
    flag = False

    for i in range(len(network_caffe.layers)):
        layer = network_caffe.layers[i]
        type_ = layer.type
        name = network_caffe._layer_names[i]

        if type_ == 'Convolution':
            weights = layer.blobs[0].data

            if not flag and not first_layer:
                flag = True
                mode_4 = unfold(weights, 0)
                _, s_, _, _ = VBMF(mode_4)
                mode_4_dim = len(np.diag(s_))
                layers_ranks[name] = (weights.shape[1], mode_4_dim)
                continue

            mode_3 = unfold(weights, 1)
            mode_4 = unfold(weights, 0)
            _, s_, _, _ = VBMF(mode_3)
            mode_3_dim = len(np.diag(s_))
            _, s_, _, _ = VBMF(mode_4)
            mode_4_dim = len(np.diag(s_))
            layers_ranks[name] = (mode_3_dim, mode_4_dim)
        elif type_ == 'InnerProduct':
            weights = layer.blobs[0].data
            _, s_, _, _ = VBMF(weights)
            dim = len(np.diag(s_))
            layers_ranks[name] = dim
            last_layer = (name, weights.shape[0])

    for key in layers_ranks:
        if key == last_layer[0]:
            layers_ranks[key] = last_layer[1]

    return layers_ranks


def proto_decomposition(network_in, network_out, layers_ranks):
    proto_in = caffe_pb2.NetParameter()

    with open(network_in, 'r') as file:
        text_format.Merge(str(file.read()), proto_in)

    proto_out = caffe_pb2.NetParameter()
    proto_out.CopyFrom(proto_in)
    proto_out.ClearField('layer')
    channel_buffer = {}

    for layer in proto_in.layer:
        if layer.type != 'Convolution' and layer.type != 'InnerProduct':
            proto_out.layer.add()
            proto_out.layer[-1].CopyFrom(layer)

            if layer.type == 'Data':
                channel_buffer[layer.top[0]] = 3
                channel_buffer[layer.top[1]] = 1
            else:
                channel_buffer[layer.top[0]] = channel_buffer[layer.bottom[0]]
        elif layer.type == 'Convolution':
            channel_buffer[layer.top[0]] = layer.convolution_param.num_output
            layer.convolution_param.ClearField('weight_filler')
            layer.convolution_param.ClearField('bias_filler')

            if layer.name not in layers_ranks:
                proto_out.layer.add()
                proto_out.layer[-1].CopyFrom(layer)
            else:
                if layers_ranks[layer.name][0] != channel_buffer[layer.bottom[0]]:
                    proto_out.layer.add()
                    lra_a_layer = proto_out.layer[-1]
                    lra_a_layer.CopyFrom(layer)
                    lra_a_layer.name += '_lra_a'
                    lra_a_layer.convolution_param.kernel_size[0] = 1
                    lra_a_layer.convolution_param.num_output = layers_ranks[layer.name][0]
                    lra_a_layer.convolution_param.ClearField('pad')
                    lra_a_layer.convolution_param.ClearField('stride')
                    lra_a_layer.top[0] = layer.name + '_lra_a'
                    channel_buffer[lra_a_layer.top[0]] = layers_ranks[layer.name][0]
                proto_out.layer.add()
                lra_b_layer = proto_out.layer[-1]
                lra_b_layer.CopyFrom(layer)
                lra_b_layer.name += '_lra_b'
                lra_b_layer.convolution_param.num_output = layers_ranks[layer.name][1]

                if layer.name + '_lra_a' in channel_buffer:
                    lra_b_layer.bottom[0] = layer.name + '_lra_a'

                if layers_ranks[layer.name][1] != channel_buffer[layer.top[0]]:
                    lra_b_layer.top[0] = layer.name + '_lra_b'
                    proto_out.layer.add()
                    lra_c_layer = proto_out.layer[-1]
                    lra_c_layer.CopyFrom(layer)
                    lra_c_layer.name += '_lra_c'
                    lra_c_layer.convolution_param.kernel_size[0] = 1
                    lra_c_layer.convolution_param.ClearField('pad')
                    lra_c_layer.convolution_param.ClearField('stride')
                    lra_c_layer.bottom[0] = layer.name + '_lra_b'
                    channel_buffer[lra_c_layer.bottom[0]] = layers_ranks[layer.name][1]
        elif layer.type == 'InnerProduct':
            channel_buffer[layer.top[0]] = layer.inner_product_param.num_output
            layer.inner_product_param.ClearField('weight_filler')
            layer.inner_product_param.ClearField('bias_filler')

            if layer.name not in layers_ranks:
                proto_out.layer.add()
                proto_out.layer[-1].CopyFrom(layer)
            else:
                proto_out.layer.add()
                svd_a_layer = proto_out.layer[-1]
                svd_a_layer.CopyFrom(layer)
                svd_a_layer.name += '_svd_a'
                svd_a_layer.inner_product_param.num_output = layers_ranks[layer.name]

                if layers_ranks[layer.name] != channel_buffer[layer.top[0]]:
                    svd_a_layer.top[0] = layer.name + '_svd_a'
                    channel_buffer[svd_a_layer.top[0]] = layers_ranks[layer.name]
                    proto_out.layer.add()
                    svd_b_layer = proto_out.layer[-1]
                    svd_b_layer.CopyFrom(layer)
                    svd_b_layer.name += '_svd_b'
                    svd_b_layer.bottom[0] = layer.name + '_svd_a'

    with open(network_out, 'w') as file:
        file.write(text_format.MessageToString(proto_out))


def weights_decomposition(network_in, weights_in, network_out, weights_out, layers_ranks):
    network_caffe = caffe.Net(network_in, weights_in, caffe_pb2.TEST)
    layers_map = dict()

    for idx in range(len(network_caffe.layers)):
        if len(network_caffe.layers[idx].blobs) > 0:
            layers_map[network_caffe._layer_names[idx]] = (network_caffe.layers[idx].type,
                                                           [blob.data for blob in network_caffe.layers[idx].blobs])

    network_caffe = caffe.Net(network_out, caffe_pb2.TEST)

    for name, (type_, blobs) in list(layers_map.items()):
        if name not in layers_ranks:
            for idx, blob in enumerate(blobs):
                update_layer(network_caffe, name, idx, blob)
        else:
            if type_ == 'Convolution':
                weight = blobs[0]
                bias = blobs[1]
                core, u = tucker_decomposition(weight, (layers_ranks[name][1], layers_ranks[name][0],
                                                        weight.shape[2], weight.shape[3]))
                lra_a = np.transpose(u[1]).reshape(u[1].shape[1], u[1].shape[0], 1, 1)
                lra_b = core
                lra_c = u[0].reshape(u[0].shape[0], u[0].shape[1], 1, 1)

                if layers_ranks[name][0] != weight.shape[1]:
                    update_layer(network_caffe, name + '_lra_a', 0, lra_a)
                    update_layer(network_caffe, name + '_lra_a', 1, np.zeros(lra_a.shape[0]))

                update_layer(network_caffe, name + '_lra_b', 0, lra_b)

                if layers_ranks[name][1] != weight.shape[0]:
                    update_layer(network_caffe, name + '_lra_b', 1, np.zeros(lra_b.shape[0]))
                    update_layer(network_caffe, name + '_lra_c', 0, lra_c)
                    update_layer(network_caffe, name + '_lra_c', 1, bias)
                else:
                    update_layer(network_caffe, name + '_lra_b', 1, bias)
            if type_ == 'InnerProduct':
                weight = blobs[0]
                bias = blobs[1]

                if layers_ranks[name] == weight.shape[0]:
                    update_layer(network_caffe, name + '_svd_a', 0, weight)
                    update_layer(network_caffe, name + '_svd_a', 1, bias)
                else:
                    u, s, v_t = np.linalg.svd(weight, full_matrices=False)
                    k = layers_ranks[name]
                    w1 = np.dot(np.diag(np.sqrt(s[0:k])), v_t[0:k, :])
                    w2 = np.dot(u[:, 0:k], np.diag(np.sqrt(s[0:k])))
                    update_layer(network_caffe, name + '_svd_a', 0, w1)
                    update_layer(network_caffe, name + '_svd_a', 1, np.zeros(k))
                    update_layer(network_caffe, name + '_svd_b', 0, w2)
                    update_layer(network_caffe, name + '_svd_b', 1, bias)

    network_caffe.save(weights_out)


def update_layer(net, layer_name, index, data):
    layer_idx = list(net._layer_names).index(layer_name)
    np.copyto(net.layers[layer_idx].blobs[index].data, data)
