import tensorflow as tf

@tf.function
def nt_xent_loss(z1: tf.Tensor, z2: tf.Tensor, temperature: float = 0.5) -> tf.Tensor:
    # z1,z2: [B,d] assumed L2-normalized
    B = tf.shape(z1)[0]
    Z = tf.concat([z1, z2], axis=0)                 # [2B,d]
    sim = tf.matmul(Z, Z, transpose_b=True)         # cosine since z are normed
    mask = tf.eye(2 * B, dtype=tf.bool)
    sim = tf.where(mask, tf.zeros_like(sim), sim)   # remove self-sim
    pos = tf.concat([tf.range(B, 2 * B), tf.range(0, B)], axis=0)
    logits = sim / temperature
    labels = tf.one_hot(pos, depth=2 * B)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)