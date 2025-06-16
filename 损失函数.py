def mse(y_true, y_pred):
    n = len(y_true)
    mse_sum = 0 
    for i in range(n):
        mse_sum+=(y_true[i]-y_pred[i])**2
    return mse_sum/n

def mae(y_true, y_pred):
    n = len(y_true)
    mae_sum = 0
    for i in range(n):
        mae_sum+=abs(y_true[i]-y_pred[i])
    return mae_sum/n

def rmse(y_true, y_pred):
    import numpy as np
    n = len(y_true)
    rmse_sum = 0
    for i in range(n):
        rmse_sum+=(y_true[i]-y_pred[i])**2
    return np.sqrt(rmse_sum)

def binary_cross_entropy(y_true, y_pred):
    """单个样本的二分类的BCE"""
    import numpy as np
    term_0 = (1-y_true)*np.log(1-y_pred)
    term_1 = y_true*np.log(y_pred)
    loss = -(term_0+term_1)
    return loss

def categorical_cross_entropy(y_true, y_pred):
    """单个样本多分类的BCE"""
    import numpy as np
    loss = -np.sum(y_true*np.log(y_pred))
    return loss

def binary_cross_entropy(y_true, y_pred):
    """单个样本的二分类的BCE"""
    import numpy as np
    term_0 = (1-y_true)*np.log(1-y_pred)
    term_1 = y_true*np.log(y_pred)
    loss = -np.mean(term_0+term_1)
    return loss

def categorical_cross_entropy(y_true, y_pred):
    """多个样本的多分类的BCE"""
    import numpy as np
    loss=-np.sum(y_true*np.log(y_pred), axis=1)
    avg_loss = np.mean(loss)
    return avg_loss