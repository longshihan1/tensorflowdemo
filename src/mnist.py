"""
学习构建一个TensorFlow模型的基本步骤，并将通过这些步骤为MNIST构建一个深度卷积神经网络。
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#加载mnist数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#为了不在建立模型的时候反复做初始化操作，我们定义两个函数用于初始化。
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#卷积和池化
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#输入变量x为一个占位符，表示mnist图像，每一张图展开成784维的向量
x = tf.placeholder(tf.float32, shape=[None, 784])

#W和b为权重和偏置变量
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#模型y=x*W+b，计算每个分类的softmax概率值
y = tf.nn.softmax(tf.matmul(x,W) + b)

#为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值。
#y_是一个2维张量，其中每一行为一个10维的one-hot向量,用于代表对应某一MNIST图片的类别。
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#为训练过程指定最小化误差用的损失函数，这里损失函数是目标类别和预测类别之间的交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#选择的优化算法来不断更新参数，这里使用最速下降法让交叉熵下降，步长为0.01。
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化创建的变量
init = tf.initialize_all_variables()

#在Session中启动模型
sess = tf.InteractiveSession()

#为初始值指定具体值（本例当中是全为零），并将其分配给每个变量,可以一次性为所有变量完成此操作。
sess.run(tf.initialize_all_variables())

#训练模型，每一步迭代，加载50个训练样本，然后执行一次train_step，并通过feed_dict将x 和 y_张量占位符用训练训练数据替代。
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#检测预测是否与真实标签匹配
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#计算我们分类的准确率均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#计算出在测试数据上的准确率，大概是91%。
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))