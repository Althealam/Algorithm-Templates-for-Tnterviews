from tensorflow.keras import layers
import tensorflow as tf

def se_block(input_tensor, ratio=16):
    """
    构建 Squeeze-and-Excitation 模块
    :param input_tensor: 输入张量 (batch_size, seq_len, channels)
    :param ratio: 压缩比例
    :return: 经过 SE 模块处理后的张量
    """
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    channels = input_tensor.shape[channel_axis] # 获取输入张量的通道数
    
    # Squeeze 操作：全局平均池化
    se_tensor = layers.GlobalAveragePooling1D()(input_tensor) # (batch_size, channels) 对输入张量在序列长度维度上进行平均池化
    se_tensor = layers.Reshape((1, channels))(se_tensor) # (batch_size, 1, channels)
    
    # Excitation 操作：全连接层
    se_tensor = layers.Dense(channels // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se_tensor) # (batch_size, 1, channels//ratio)
    se_tensor = layers.Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se_tensor) # (batch_size, 1, channels)
    
    # Scale 操作：将注意力权重与输入特征相乘
    if tf.keras.backend.image_data_format() == "channels_first":
        se_tensor = layers.Permute((3, 1, 2))(se_tensor)  # (batch_size, 1, channels)
    x = layers.Multiply()([input_tensor, se_tensor]) # (batch_size, seq_len, channels)
    return x