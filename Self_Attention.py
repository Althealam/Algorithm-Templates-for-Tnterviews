import tensorflow as tf

def self_attention(inputs, d_k, mask=None):
    """
    自注意力机制
    :param inputs: 输入张量 (batch_size, seq_len, d_model)
    :param d_k: key和value向量的维度
    :param mask: 掩码张量，可以屏蔽某些位置的注意力计算
    
    :return 自注意力输出，形状为(batch_size, seq_len, d_model)
    """
    batch_size, seq_len, d_model = inputs.shape

    W_q = tf.keras.layers.Dense(d_k)
    W_k = tf.keras.layers.Dense(d_k)
    W_v = tf.keras.layers.Dense(d_k)

    # 计算q, k, v
    q = W_q(inputs) # (batch_size, seq_len, d_k)
    k = W_k(inputs) # (batch_size, seq_len, d_k)
    v = W_v(inputs) # (batch_size, seq_len, d_k)

    # 计算注意力分数
    attention_score = tf.matmul(q, k, transpose_b=True)/tf.math.sqrt(tf.cast(d_k, tf.float32)) # (batch_size, seq_len, seq_len)

    if mask is not None:
        attention_score += (mask*-1e9)

    # 利用softmax得到注意力权重
    attention_weights = tf.nn.softmax(attention_score, axis=-1) # (batch_size, seq_len, seq_len)

    # 计算注意力输出
    output = tf.matmul(attention_weights, v) # (batch_size, seq_len, d_model)
    return output