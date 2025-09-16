import math
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Activation, AveragePooling2D, Conv1D, Conv2D,
    DepthwiseConv2D, BatchNormalization, Add, Lambda, Permute, LayerNormalization,
    Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2
from tensorflow.keras.constraints import max_norm

SEED = 1

def _mha_block(x, key_dim=8, num_heads=2, dropout=0.5, vanilla=True):
    x_in = x
    x = LayerNormalization(epsilon=1e-6)(x)
    if vanilla:
        x = tf.keras.layers.MultiHeadAttention(key_dim=key_dim, num_heads=num_heads, dropout=dropout)(x, x)
    else:
        x = _MultiHeadAttentionLSA(key_dim=key_dim, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(0.3, seed=SEED)(x)
    return Add()([x_in, x])

class _MultiHeadAttentionLSA(tf.keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        scores = tf.einsum(self._dot_product_equation, key, query)
        scores = self._masked_softmax(scores, attention_mask)
        scores = self._dropout_layer(scores, training=training)
        out = tf.einsum(self._combine_equation, scores, value)
        return out, scores

def _attention_block(x, kind: str):
    if kind in ("mha", "mhla"):
        if len(x.shape) > 3:  # [B,T,C,?] -> [B,T,C]
            x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], -1))
        return _mha_block(x, vanilla=(kind == "mha"))
    return x  # no-op for unsupported kinds here

def _conv_stem(x, F1=16, kern=64, pool=7, D=2, in_ch=22, wd=0.009, maxn=0.6, p=0.3):
    F2 = F1 * D
    x = Conv2D(F1, (kern, 1), padding="same", data_format="channels_last",
               kernel_regularizer=L2(wd),
               kernel_constraint=max_norm(maxn, axis=[0, 1, 2]),
               use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = DepthwiseConv2D((1, in_ch), depth_multiplier=D, data_format="channels_last",
                        depthwise_regularizer=L2(wd),
                        depthwise_constraint=max_norm(maxn, axis=[0, 1, 2]),
                        use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((8, 1), data_format="channels_last")(x)
    x = Dropout(p, seed=SEED)(x)
    x = Conv2D(F2, (16, 1), padding="same", data_format="channels_last",
               kernel_regularizer=L2(wd),
               kernel_constraint=max_norm(maxn, axis=[0, 1, 2]),
               use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((pool, 1), data_format="channels_last")(x)
    x = Dropout(p, seed=SEED)(x)
    return x  # [B,T',F2,1]

def _tcn_block(x, input_dim, depth, kernel_size, filters, p=0.3, wd=0.009, maxn=0.6, act="elu"):
    def conv(a, dil):
        a = Conv1D(filters, kernel_size=kernel_size, dilation_rate=dil, activation="linear",
                   kernel_regularizer=L2(wd),
                   kernel_constraint=max_norm(maxn, axis=[0, 1]),
                   padding="causal", kernel_initializer="he_uniform")(a)
        a = BatchNormalization()(a); a = Activation(act)(a); a = Dropout(p, seed=SEED)(a)
        a = Conv1D(filters, kernel_size=kernel_size, dilation_rate=dil, activation="linear",
                   kernel_regularizer=L2(wd),
                   kernel_constraint=max_norm(maxn, axis=[0, 1]),
                   padding="causal", kernel_initializer="he_uniform")(a)
        a = BatchNormalization()(a); a = Activation(act)(a); a = Dropout(p, seed=SEED)(a)
        return a
    y = conv(x, 1)
    skip = x if input_dim == filters else Conv1D(filters, 1, padding="same")(x)
    y = Activation(act)(Add()([y, skip]))
    for i in range(depth - 1):
        z = conv(y, 2 ** (i + 1))
        y = Activation(act)(Add()([z, y]))
    return y

def build_atcnet(
    n_classes: int,
    *,
    in_chans: int = 22,
    in_samples: int = 1000,
    n_windows: int = 5,
    attention: str = "mha",
    eegn_F1: int = 16,
    eegn_D: int = 2,
    eegn_kernel: int = 64,
    eegn_pool: int = 7,
    eegn_dropout: float = 0.3,
    tcn_depth: int = 2,
    tcn_kernel: int = 4,
    tcn_filters: int = 32,
    tcn_dropout: float = 0.3,
    tcn_activation: str = "elu",
    fuse: str = "average",
    from_logits: bool = False,
    return_ssl_feat: bool = False,
) -> Model:
    inp = Input(shape=(1, in_chans, in_samples))
    x = Permute((3, 2, 1))(inp)  # [B,T,C,1]

    x = _conv_stem(
        x, F1=eegn_F1, kern=eegn_kernel, pool=eegn_pool, D=eegn_D,
        in_ch=in_chans, wd=0.009, maxn=0.6, p=eegn_dropout
    )
    x = Lambda(lambda z: z[:, :, -1, :])(x)  # [B,T',F2]
    F2 = eegn_F1 * eegn_D

    feats_or_logits = []
    per_win_feats = []

    L = tf.shape(x)[1]
    nW = tf.cast(n_windows, tf.int32)
    for i in range(max(n_windows, 1)):
        if n_windows == 1:
            xi = x
        else:
            st = i
            en = L - nW + i + 1
            xi = x[:, st:en, :]

        if attention in ("mha", "mhla"):
            xi = _attention_block(xi, attention)

        xi = _tcn_block(
            xi, input_dim=F2, depth=tcn_depth, kernel_size=tcn_kernel,
            filters=tcn_filters, p=tcn_dropout, wd=0.009, maxn=0.6, act=tcn_activation
        )
        xi = Lambda(lambda z: z[:, -1, :])(xi)  # [B, tcn_filters]
        per_win_feats.append(xi)

        if fuse == "average":
            logits_i = Dense(n_classes, kernel_regularizer=L2(0.5))(xi)
            feats_or_logits.append(logits_i)
        elif fuse == "concat":
            feats_or_logits = xi if i == 0 else Concatenate(axis=-1)([feats_or_logits, xi])
        else:
            raise ValueError(f"Unsupported fuse='{fuse}'.")

    if fuse == "average":
        out_pre = feats_or_logits[0] if len(feats_or_logits) == 1 else tf.keras.layers.Average()(feats_or_logits)
    else:
        out_pre = Dense(n_classes, kernel_regularizer=L2(0.5))(feats_or_logits)

    out = Activation("linear" if from_logits else "softmax", name="output")(out_pre)

    if not return_ssl_feat:
        return Model(inputs=inp, outputs=out, name="ATCNet")

    ssl_feat = per_win_feats[0] if len(per_win_feats) == 1 else tf.keras.layers.Average(name="ssl_feat")(per_win_feats)
    return Model(inputs=inp, outputs=[out, ssl_feat], name="ATCNet")