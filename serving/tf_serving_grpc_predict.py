from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import tensorflow as tf
import numpy as np

tf.compat.v1.app.flags.DEFINE_string('server', '0.0.0.0:8500', 'PredictionService host:port')
FLAGS = tf.compat.v1.app.flags.FLAGS

feature_dict = {}


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# feature_dict['uid'] = _bytes_feature(bytes('b\'1023\'', encoding='utf-8'))
feature_dict['uid'] = _int64_feature(102299)
feature_dict['1039_7_order_cnt'] = _int64_feature(90)
example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
serialized = example_proto.SerializeToString()
channal = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channal)
requset = predict_pb2.PredictRequest()
requset.model_spec.name = 'rec_model'
requset.model_spec.signature_name = 'predict'

example = tf.train.Example()
# example.features.feature['sku_sn'].bytes_list.value.extend([bytes('83923', 'utf-8')])
example.features.feature['uid'].int64_list.value.extend([568810000453276939])
example.features.feature['sku_sn'].int64_list.value.extend([2406096])
# example.features.feature["1039_7_order_cnt"].int64_list.value.extend([7])
b = example.SerializeToString()
a = tf.make_tensor_proto(example.SerializeToString(), dtype=tf.string)

requset.inputs['examples'].CopyFrom(tf.make_tensor_proto(tf.constant([b], dtype=tf.string)))

response = stub.Predict(requset, 10.0)

print(response)

'''
# 获取stub
channel = implementations.insecure_channel('localhost', 8500)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel._channel)

# 模型签名
request = predict_pb2.PredictRequest()
request.model_spec.name = 'model_other'
# request.model_spec.version = 'latest'
request.model_spec.signature_name = 'predict'

# 构造入参
x_data = [[13, 45, 13, 13, 49, 1, 49, 196, 594, 905, 48, 231, 318, 712, 1003, 477, 259, 291, 287, 161, 65, 62, 82, 68, 2, 10]]
drop_out = 1
sequence_length = [26]
request.inputs['input'].CopyFrom(tf.make_tensor_proto(x_data, dtype=tf.int32))
request.inputs['sequence_length'].CopyFrom(tf.make_tensor_proto(sequence_length, dtype=tf.int32))
request.inputs['drop_out'].CopyFrom(tf.make_tensor_proto(drop_out, dtype=tf.float32))

#  返回CRF结果，输出发射概率矩阵和状态转移概率矩阵
result = stub.Predict(request, 10.0)  # 10 secs timeout

print(result)
'''
