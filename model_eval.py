import tensorflow as tf
import time
from features import test_features

# serving
imported = tf.saved_model.load('C:/work/My_model/serving/saved_model_dir/1684462413')
print(imported)


def predict(x):
    example = tf.train.Example()
    example.features.feature["product_id"].int64_list.value.extend([x])
    print(example)
    return imported.signatures["predict"](
        examples=tf.constant([example.SerializeToString()]))


'''
if dtype == "object":
    value = bytes(value, "utf-8")
        example.features.feature[colName].bytes_list.value.extend([value])
elif dtype == "float":
    example.features.feature[colName].float_list.value.extend([value])
elif dtype == "int":
    example.features.feature[colName].int64_list.value.extend([value])
'''

feature_description = {}

feature_description.update({
    f: tf.io.FixedLenFeature([], tf.int64, 0)
    for f in test_features
})
'''
feature_description.update({
    f: tf.io.FixedLenFeature([], tf.string, "")
    for f in str_id_features
})
'''
# feature_description.update({
#     f: tf.io.FixedLenFeature([], tf.int64, 0)
#     for f in long_id_features
# })
# feature_description.update({
#     f: tf.io.VarLenFeature(tf.string)
#     for f in list_features
# })

feature_description['application'] = tf.io.FixedLenFeature([], tf.int64, 0)


def parser(record):
    read_data = tf.io.parse_example(serialized=record, features=feature_description)
    return read_data


# tmp = '/home/jovyan/data-vol-1/user_feature_tfrecord_new/part-r-00001'
tmp = 'dataset/part-r-1.tfrecords'
ds = tf.data.TFRecordDataset(tmp)
ds = ds.repeat(1).batch(1)
ds = ds.map(lambda x: parser(x))

start_time = time.time()
cnt = 0
right = 0
pos = 1
neg = 0
pos_right = 1
example_str = []
store_feature = {}
transformation_time = time.time()
for feature_dict in ds:
    example = tf.train.Example()
    # for feature in long_id_features:
    #     example.features.feature[feature].int64_list.value.extend([feature_dict[feature].numpy()])

    for feature in test_features:
        print(feature)
        example.features.feature[feature].int64_list.value.extend([feature_dict[feature].numpy()])

    # for feature in str_id_features:
    #    value = bytes(str(feature_dict[feature].numpy()), "utf-8")
    #    example.features.feature[feature].bytes_list.value.extend([value])
    # for feature in list_features:
    #     dense_tensor = tf.sparse.to_dense(feature_dict[feature])
    #     value = bytes(str(tf.squeeze(dense_tensor).numpy()), "utf-8")
    #     example.features.feature[feature].bytes_list.value.extend([value])

    result = imported.signatures["serving_default"](
        examples=tf.constant([example.SerializeToString()]))
    print(result)
    exit()
    click = feature_dict['click'].numpy()
    uid = feature_dict['uid'].numpy()
    sku_sn = feature_dict['sku_sn'].numpy()
    class_id = result['class_ids'].numpy()[0]
    if class_id == 0:
        pos += 1
    if click == 0 and class_id == 0:
        pos_right += 1
    if class_id == click:
        right += 1
    cnt += 1
    print(right/cnt)
    example_str.append(example.SerializeToString())
transformation_end = time.time()
print("Transform const {}".format(transformation_end - transformation_time))
# click = feature_dict['click'].numpy()
predict_start = time.time()
result = imported.signatures["predict"](
    examples=tf.constant(example_str))
predict_end = time.time()
print("predict need {}".format(predict_end - predict_start))
print(len(result['logistic'].numpy()))
'''
if predict_id == click:
    right += 1
cnt += 1
test_end_time = time.time()
print("total number is {}".format(cnt))
print("right prediction number is {}".format(right))
print("correct rate is {}".format(float(right/cnt)))
print("test cost time {}".format(test_end_time - start_time))
'''