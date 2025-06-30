import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

import tensorflow as tf
from tensorflow.keras import layers


def multi_head_attention(inputs, d_model, num_heads, mask=None, dropout=0.1):
    assert d_model%num_heads ==0
    d_k = d_model//num_heads

    # 投影层
    W_q = tf.keras.layers.Dense(d_model)
    W_k = tf.keras.layers.Dense(d_model)
    W_v = tf.keras.layers.Dense(d_model)

    batch_size, seq_len, _ = inputs.shape

    q = W_q(inputs) # (batch_size, seq_len, d_model)
    k = W_k(inputs) # (batch_size, seq_len, d_model)
    v = W_v(inputs) # (batch_size, seq_len, d_model)

    def split_heads(x):
        """
        input: (batch_size, seq_len, d_model)
        return: (batch_size, num_heads, seq_len, d_k)
        """
        x = tf.reshape(x [batch_size, -1, num_heads, d_k])
        return tf.transpose(x, perm=[0,2,1,3])
    
    q = split_heads(q) # (batch_size, num_heads, seq_len, d_k)
    k = split_heads(k) # (batch_size, num_heads, seq_len, d_k)
    v = split_heads(v) # (batch_size, num_heads, seq_len, d_k)

    scores = tf.matmul(q, k, transpose=True) # (batch_size, num_heads, seq_len, seq_len)
    scores = scores/tf.math.sqrt(tf.cast(d_k, tf.float32))

    if mask is not None:
        scores+=(mask*-1e9)

    attention_weights = tf.nn.softmax(scores, axis=-1) # (batch_size, num_heads, seq_len, seq_len)
    attention_weights = tf.keras.layers.Dropout(dropout)(attention_weights)

    output = tf.matmul(attention_weights, v) # (batch_size, num_heads, seq_len, d_k)

    output = tf.transpose(output, [0, 2, 1, 3]) #(batch_size,seq_len, num_heads, d_k)
    output = tf.reshape(output, [batch_size, seq_len, d_model]) # (batch_size, seq_len, d_model)

    return output

    

# class MHA(layers.Layer):
#     def __init__(self, d_model, num_heads):
#         super(MHA, self).__init__()
#         assert d_model%num_heads == 0
#         self.num_heads = num_heads
#         self.depth = d_model//num_heads
#         self.d_model = d_model
        
#         self.wq = layers.Dense(d_model)
#         self.wk = layers.Dense(d_model)
#         self.wv = layers.Dense(d_model)
        
#         self.dense = layers.Dense(d_model)
    
#     def split_heads(self, x, batch_size):
#         """
#         x: (batch_size, seq_len, d_model)
#         : return (batch_size, num_heads, seq_len, depth)
#         """
#         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth)
#         return tf.transpose(x, perm=[0,2,1,3])
        
#     def call(self, q, k, v):
#         batch_size = tf.shape(q)[0]
        
#         q = self.wq(q) # (batch_size, seq_len_q, d_model)
#         k = self.wk(k) # (batch_size, seq_len_k, d_model)
#         v = self.wv(v) # (batch_size, seq_len_v, d_model)
        
#         q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
#         k = self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_k, depth)
#         v = self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, depth)
        
#         # 计算点积注意力
#         matmul_qk = tf.matmul(q, k, transpose_b=True) # (batch_size, heads, seq_len_q, seq_len_k)
        
#         dk = tf.cast(tf.shape(k)[-1], tf.float32)
#         scaled_attention_logits = matmul_qk/tf.math.sqrt(dk) # (batch_size, heads, seq_len_q, seq_len_k)
        
#         attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (batch_size, heads, seq_len_q, seq_len_k)
        
#         output = tf.matmul(attention_weights, v) # (batch_size, heads, seq_len_q, depth)
        
#         # 多头结果合并
#         output = tf.transpose(output, perm=[0,2,1,3]) # (batch_size, seq_len_q, heads, depth)
#         concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        
#         # 通过最后的线形层
#         output= self.dense(concat_attention) # (batch_size, seq_len_q, d_model)
        
#         return output 

