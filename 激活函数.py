## Sigmoid函数
import math
def sigmoid(x):
    return 1/(1+math.exp(-x))

## Tanh函数
import math
def tanh(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

## ReLU函数
def relu(x):
    return max(0, x)

## Leaky ReLU函数
def leaky_relu(x, alpha):
    if x>0:
        return x
    else:
        return alpha*x
    
## Softmax函数
import math
def softmax(x):
    exp_x=[math.exp(xi) for xi in x]
    sum_exp_x = sum(exp_x)
    return [xi/sum_exp_x for xi in exp_x]