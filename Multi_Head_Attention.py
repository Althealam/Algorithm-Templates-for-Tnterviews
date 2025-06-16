import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout


def scaled_dot_product_attention(q, k, v, mask=None, dropout_rate=0.1):
    """点积注意力"""
    # 计算注意力得分
    # q: (batch_size, num_heads, seq_len_q, depth)
    # k: (batch_size, num_heads, seq_len_k, depth) -> (batch_size, num_heads, depth, seq_len_k)
    matmul_qk = tf.matmul(q, k, tranpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)

    # 转换dk的数据类型：取k的最后一个维度为depth，转换为float32
    dk=tf.cast(tf.shape(k)[-1], tf.float32) # depth
    # 缩放注意力得分
    # (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention_logits = matmul_qk/tf.math.sqrt(dk) # qk^T/(dk)^(1/2)

    # 应用mask
    if mask is not None:
        scaled_attention_logits+=(mask*1e-9) # 将mask的值设置为负无穷，softmax后为0

    # 应用softmax获取注意力权重
    # (batch_size, num_heads, seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # 进行Dropout
    attention_weights = Dropout(rate=dropout_rate)(attention_weights)

    # 和v相乘计算输出
    # attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
    # v:  (batch_size, num_heads, seq_len_k, depth)
    output= tf.matmul(attention_weights, v) # (batch_size, num_heads, seq_len_q, depth)

    return output, attention_weights



class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model # 模型的维度
        self.num_heads = num_heads # 头的数量

        # 确保模型的维度除以头的数量是整数
        assert self.d_model%self.num_heads==0

        self.depth=self.d_model/self.num_heads # 每个头的维度

        # 初始化qkv的线性矩阵
        self.wq=Dense(d_model)
        self.wk=Dense(d_model)
        self.wv=Dense(d_model)

        # 初始化Dropout层
        self.dropout=Dropout(dropout_rate)

    def split_heads(self, x, batch_size):
        """
        分割头 
        x的维度是(batch_size, seq_len, d_model)，分割后x的维度是(batch_size, num_heads, seq_len, d_model//num_heads)
        """
        # 对最后一个维度进行分割
        x=tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        # (batch_size, seq_len, num_heads, d_model/num_heads)

        # 转换维度
        x=tf.transpose(x, perm=[0, 2, 1, 3])

        return x
    
    def call(self, q, k, v, mask=None, training=None):
        batch_size = tf.shape(q)[0] # 获取数据批次

        # 线形变换
        q=self.wq(q) # (batch_size, seq_len_q, d_model)
        k=self.wk(k) # (batch_size, seq_len_k, d_model)
        v=self.wv(v) # (batch_size, seq_len_v, d_model)

        # 分割头
        q=self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, d_model//num_heads)
        k=self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_k, d_model//num_heads)
        v=self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, d_model//num_heads)
        # 最后变为(batch_size, num_heads, seq_len, depth)

        # 缩放点积注意力
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        # scaled_attention: (batch_size, num_heads, seq_len_k, depth)
        # attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)

        # 转置点积注意力
        # (batch_size, seq_len_k, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # 合并多头
        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # 将num_heads和depth合并在一起

        # 最后线性层
        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights