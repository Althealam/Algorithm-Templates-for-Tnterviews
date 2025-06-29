# ============= 梯度下降 =================
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    def update(self, params, gradients):
        updated_params = []
        for p, g in zip(params, gradients):
            # 核心公式：新参数 = 参数 - 学习率 * 梯度
            new_p = p - self.learning_rate * g
            updated_params.append(new_p)
        return updated_params
    

# ============= 随机梯度下降 =================
import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, gradients):
        updated_params = []
        for p, g in zip(params, gradients):
            updated_p = p-self.learning_rate*g
            updated_params.append(updated_p)
        return updated_params

if __name__=='__main__':
    x = np.array([1, 2, 3 ,4, 5], dtype=np.float32)
    y = np.array([2, 4, 6, 8, 10], dtype=np.float32)

    np.random.seed(42)
    w = np.random.randn(1, 1) # 权重
    b = np.random.randn(1, 1) # 偏置
    params = [w, b]

    # 定义模型
    def model(x, params):
        w, b = params
        return np.dot(x, w)+b
    
    # 定义损失函数
    def loss(y_pred, y_true):
        return np.mean((y_pred-y_true)**2)

    # 定义梯度计算函数
    def compute_gradients(x, y_true, y_pred, params):
        w, b = params
        m = len(x)
        dw = (1/m)*np.dot(x.T, (y_pred-y_true)) # 对w的梯度
        db = (1/m)*np.sum(y_pred-y_true)
        return [dw, db]

    optimizer = SGD(learning_rate=0.01)

    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        for i in range(len(x)):
            # 随机选择单个样本
            x_sample = x[i:i+1]
            y_sample = y[i:i+1]

            # 前向传播
            y_pred = model(x_sample, y_sample)

            # 计算损失和梯度
            current_loss = loss(y_pred, y_sample)
            gradients = compute_gradients(x_sample, y_sample, y_pred, params)

            # 更新参数
            params = optimizer.update(params, gradients)
        
        # 打印损失
        if epoch%10==0:
            y_pred = model(x, params)
            total_loss = loss(y_pred, y)
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}, w: {w[0][0]:.4f}, b: {b[0][0]:.4f}")


# ============= Adam优化器 ==============
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps = 1e-8):
        self.learning_rate = learning_rate # 学习率
        self.beta1 = beta1 # 一阶矩（动量）衰减系数
        self.beta2 = beta2 # 二阶矩（方差）衰减系数
        self.epo = eps # 防止除以0的极小值
        self.t=0 # 迭代次数
        self.m=[] # 一阶矩（均值）
        self.v=[] # 二阶矩（方差）

    def update(self, params, gradients):
        self.t+=1 # 迭代次数+1

        # 初始化一阶矩和二阶矩
        if not self.m:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_list(p) for p in params]

        updated_params=[]
        for i, (p, g) in enumerate(zip(params, gradients)):
            # 计算一阶矩
            self.m[i]=self.beta1*self.m[i]+(1-self.beta1)
            # 计算二阶矩（方差项）
            self.v[i]=self.beta2*self.v[i]+(1-self.beta2)
            # 偏差修正
            m_hat=self.m[i]/(1-self.beta1**self.t)
            v_hat=self.v[i]/(1-self.beta2**self.t)
            # 更新参数
            updated_p = p-self.learning_rate*m_hat/(np.sqrt(v_hat)+self.eps)
            updated_params.append(updated_p)
        return updated_params
