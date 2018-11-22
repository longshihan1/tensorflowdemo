#输入为x和y，x为[0,1)的随机数，y=0.1∗x+0.3。构造线性模型为y = Weights * x + biases，Weights和biases就是要训练的值。
import tensorflow as tf
import numpy as np

#creat data
x_data = np.random.rand(100).astype(np.float32)     ##输入值[0，1)之间的随机数
y_data = x_data * 0.1 + 0.3     ##预测值

###creat tensorflow structure strat###
# 构造要拟合的线性模型
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0)) #生成（-1,1]的一维随机数   
biases = tf.Variable(tf.zeros([1]))     #生成一维的0
y = Weights * x_data + biases   

# 定义损失函数和训练方法
loss = tf.reduce_mean(tf.square(y-y_data))   ##最小化方差    
optimizer = tf.train.GradientDescentOptimizer(0.5)#梯度下降法优化器
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()
###creat tensorflow structure end###

# 启动
sess = tf.Session()
sess.run(init)

# 训练拟合，每一步训练队Weights和biases进行更新
for step in range(301):
    sess.run(train)
    if step % 20 == 0:  
            print(step,sess.run(Weights),sess.run(biases))  ##每20步输出一下W和b

# 得到最优拟合结果 W接近于0.1，b接近于0.3