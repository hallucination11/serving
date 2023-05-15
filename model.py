import collections

try:
    from tensorflow.python.ops.init_ops_v2 import Zeros, glorot_normal
except ImportError:
    from tensorflow.python.ops.init_ops import Zeros, glorot_normal_initializer as glorot_normal
from utils import *


class Model(collections.namedtuple("Model", ["model_name",
                                             'model_dir', 'embedding_upload_hook', 'high_param'])):
    def __new__(cls,
                model_name,
                model_dir,
                embedding_upload_hook=None,
                high_param=None
                ):
        return super(Model, cls).__new__(
            cls,
            model_name,
            model_dir,
            embedding_upload_hook,
            high_param
        )

    def get_model_fn(self):
        def model_fn(features, labels, mode, params):
            pass

        return model_fn

    def get_estimator(self):
        estimator = tf.estimator.Estimator(
            model_dir=self.model_dir,
            model_fn=self.get_model_fn(),
            params={}
        )

        # add gauc

        return estimator


# recall
class DSSM(Model):
    def get_model_fn(self):

        def model_fn(features, labels, mode, params):

            # sampler_config
            sampler_config = params['sampler_config']
            item_name = sampler_config.item_name
            # train_inbatch_counter = Counter(features[item_name])
            # mast be a tensor
            # train_inbatch_counter = Counter(item_list)
            self.embedding_upload_hook.item = features[item_name]
            feature_embeddings = []
            item_embeddings = []
            feature_square_embeddings = []

            all_features = list(params['feature_columns'].keys())
            for feature in ['uid', 'item', 'gender', 'bal']:
                feature_emb = tf.compat.v1.feature_column.input_layer(features, params['feature_columns'][feature])
                if feature != 'item':
                    feature_embeddings.append(feature_emb)
                    feature_square_embeddings.append(tf.square(feature_emb))
                else:
                    item_embeddings.append(feature_emb)

            user_net = tf.concat(feature_embeddings, axis=1, name='user')
            item_net = tf.concat(item_embeddings, axis=1, name='item')

            for unit in params['hidden_units']:
                user_net = tf.compat.v1.layers.dense(user_net, units=unit, activation=tf.nn.relu)
                user_net = tf.compat.v1.layers.batch_normalization(user_net)
                user_net = tf.compat.v1.layers.dropout(user_net)
                user_net = tf.nn.l2_normalize(user_net)

            for unit in params['hidden_units']:
                item_net = tf.compat.v1.layers.dense(item_net, units=unit, activation=tf.nn.relu)
                item_net = tf.compat.v1.layers.batch_normalization(item_net)
                item_net = tf.compat.v1.layers.dropout(item_net)
                item_net = tf.nn.l2_normalize(item_net)

            if self.high_param['loss_type'] == 'sigmoid':
                dot = tf.reduce_sum(tf.multiply(user_net, item_net), axis=1, keepdims=True) / float(
                    params['temperature'])
                logits = tf.sigmoid(dot)
            if self.high_param['loss_type'] == 'softmax':
                return None

            if mode == tf.estimator.ModeKeys.PREDICT:
                approval_pred = tf.argmax(logits, axis=-1)
                predictions = {"approval_pred": approval_pred,
                               "user_emb": user_net,
                               "probability": logits}
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            approval_label = tf.reshape(labels['approval'], shape=[-1, 1])
            loss = tf.compat.v1.losses.log_loss(approval_label, logits)

            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
                train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

            if mode == tf.estimator.ModeKeys.EVAL:
                approval_pred_label = tf.argmax(logits, axis=-1)
                approval_auc = tf.compat.v1.metrics.auc(labels=approval_label, predictions=approval_pred_label)
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'approval_auc': approval_auc})

        # self.embedding_upload_hook.id_list_embedding = id_list_embedding

        return model_fn

    def get_estimator(self):
        # 商品id类特征
        def get_categorical_hash_bucket_column(key, hash_bucket_size, dimension, dtype, initializer):
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                key, hash_bucket_size=hash_bucket_size, dtype=dtype
            )
            return tf.feature_column.embedding_column(categorical_column, dimension=dimension,
                                                      initializer=tf.compat.v1.keras.initializers.random_normal(
                                                          mean=0.0, stddev=0.0001, seed=2020)) \
                if initializer == 'random_normal' else tf.feature_column.embedding_column(
                categorical_column, dimension=dimension)

        # 连续值类特征（差异较为明显）
        def get_bucketized_column(key, boundaries, dimension, initializer):
            bucketized_column = tf.feature_column.bucketized_column(
                tf.feature_column.numeric_column(key), boundaries)
            return tf.feature_column.embedding_column(bucketized_column, dimension=dimension,
                                                      initializer=tf.compat.v1.keras.initializers.random_normal(
                                                          mean=0.0, stddev=0.0001,
                                                          seed=2020)) \
                if initializer == 'random_normal' else tf.feature_column.embedding_column(
                bucketized_column, dimension=dimension)

        long_id_feature_columns = {}

        cnt_feature_columns = {
            "uid": get_categorical_hash_bucket_column("uid", hash_bucket_size=2000, dimension=6, dtype=tf.int64,
                                                      initializer=self.high_param['embedding_initializer']),
            "item": get_categorical_hash_bucket_column("item", hash_bucket_size=100, dimension=3, dtype=tf.int64,
                                                       initializer=self.high_param['embedding_initializer']),
            "bal": get_bucketized_column("bal", boundaries=[10002.0, 14158.35, 18489.0, 23177.0, 27839.8, 32521.5,
                                                            36666.7, 41386.9, 45919.6, 50264.55, 54345.0], dimension=4,
                                         initializer=self.high_param['embedding_initializer']),
            "gender": get_categorical_hash_bucket_column("gender", hash_bucket_size=2, dimension=1, dtype=tf.int64,
                                                         initializer=self.high_param['embedding_initializer'])
        }

        all_feature_column = {}
        all_feature_column.update(long_id_feature_columns)
        all_feature_column.update(cnt_feature_columns)

        weight_column = tf.feature_column.numeric_column('weight')

        hidden_layers = [256, 128]

        num_experts = 3

        task_names = ("ctr", "ctcvr", "ctvoi")

        task_types = ("binary", "binary", "binary")

        lamda = 1

        sampler_config = NegativeSampler(sampler='inbatch', num_sampled=2, item_name='item', item_count=[])

        estimator = tf.estimator.Estimator(
            model_dir=self.model_dir,
            model_fn=self.get_model_fn(),
            params={
                'hidden_units': hidden_layers,
                'feature_columns': all_feature_column,
                'weight_column': weight_column,
                'lamda': lamda,
                'num_experts': num_experts,
                'task_names': task_names,
                'task_types': task_types,
                'gate_dnn_hidden_units': [10],
                'tower_dnn_hidden_units': 64,
                'sampler_config': sampler_config,
                'temperature': self.high_param['temperature']
            })

        return estimator

