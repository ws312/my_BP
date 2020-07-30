import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
from Learn import input_data
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import time
import os

'''
L2正则化，Dropout，
'''
if not os.path.exists('result'):
    os.mkdir('result')
if not os.path.exists("log"):
    os.mkdir("log")
LOG_train = open(os.path.join("log", 'log_train.txt'), 'w')  # 训练日志
LOG_test = open(os.path.join("log", 'log_test.txt'), 'w')  # 训练日志
def log_train(out_str):
    LOG_train.write(out_str + '\n')
    LOG_train.flush()  # 清空缓存区
    print(out_str)
def log_test(out_str):
    LOG_test.write(out_str + '\n')
    LOG_test.flush()  # 清空缓存区
    print(out_str)


class my_BP(object):

    path1 = r'3.png'
    path2 = r'5.png'
    path3 = r'8.png'

    def __init__(self):
        self.tf_graph = tf.Graph()
        self.max_steps = 1000
        self.batch_size = 100  # 迷你批次的大小100个样本，100个
        self.lanmeda = 0.0001  # 定义调整项L2权值衰减系数
        self.keep_prob = 0.90
        self.learning_rate = 0.001
        self.log_dir = "log" #存放日志文件路径

    def load_datasets(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        X_train = mnist.train.images  # 取出训练样本
        y_train = mnist.train.labels  # 10维的one-hot向量
        X_validation = mnist.validation.images  # 取出验证样本
        y_validation = mnist.validation.labels
        X_test = mnist.test.images  # 取出测试样本
        y_test = mnist.test.labels
        return X_train, y_train, X_validation, y_validation, X_test, y_test, mnist

    '''创建命名空间，用summary.py中一些数据汇总函数进行数据汇总'''
    def variable_summaries(self,var):
        with tf.name_scope("summaries"):
            # 汇总var数据的平均值,并将标签设为mean
            mean = tf.reduce_mean(var)
            tf.summary.scalar("mean", mean)
            # 汇总var数据的方差值
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar("stddev", stddev)
            # 汇总var数据的最大值
            tf.summary.scalar("max", tf.reduce_max(var))
            # 汇总var数据的最小值
            tf.summary.scalar("min", tf.reduce_min(var))
            # 将var数据汇总为直方图的形式
            '''DISTRIBUTIONS'''
            tf.summary.histogram("histogram", var)

    '''创建两层MLP网络'''
    def create_layer(self,input_tensor, input_num, output_num, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope("weights"):
                weights = tf.Variable(tf.truncated_normal([input_num, output_num], stddev=0.1,mean=0.0,seed=1))
                regularizer = tf.contrib.layers.l2_regularizer(self.lanmeda)  # 计算L2正则化损失函数
                tf.add_to_collection("regularizers",regularizer(weights))
                self.variable_summaries(weights)
            with tf.name_scope("biases"):
                biases = tf.Variable(tf.constant(0.1, shape=[output_num]))
                self.variable_summaries(biases)
            with tf.name_scope("WX_add_b"):
                pre_activate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram("pre_activations", pre_activate)
            # 计算激活后的线性变换的结果，并通过histogram()函数汇总为直方图数据
            activations = act(pre_activate, name="activation")
            tf.summary.histogram("activations", activations)
        return activations

    def build_model(self):
        # 输入层
        x = tf.placeholder(tf.float32, [None, 784], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, 10], name="y-input")
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        # 隐藏层
        hidden_1 = self.create_layer(x, 784, 500, "input_to_hidden", act=tf.nn.relu)
        # 输出层
        y = self.create_layer(hidden_1, 500, 10, "hidden_to_out", act=tf.identity)#激活函数用全等映射，即不使用softmax

        '''IMAGES'''
        with tf.name_scope("input_reshape"):
            image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])#一维数据变为28x28图片
            tf.summary.image("input", image_shaped_input, max_outputs=10)  # 图片汇总数目最大10张

        '''SCALARS'''
        with tf.name_scope("loss"):
            cross = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
            cross_entropy = tf.reduce_mean(cross)  # 平均损失
            regularization=tf.add_n(tf.get_collection("regularizers"))
            loss = cross_entropy+ regularization
            tf.summary.scalar("cross_entropy_scalar", loss)  # 采用标量形式汇总平均损失

        '''损失优化'''
        with tf.name_scope("train"):
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        '''SCALARS'''
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy_scalar", accuracy)
        # 使用merge_all()函数直接获取所有汇总操作的数据
        merged = tf.summary.merge_all()
        return x,y,y_,keep_prob,loss,train_step,accuracy,merged

    def train(self):
        X_train, y_train, X_validation, y_validation, X_test, y_test, mnist=self.load_datasets()
        x, y, y_,keep_prob, loss, train_step, accuracy, merged=self.build_model()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            # 生成写日志的writer，将当前TensorFlow计算图写入日志，在不同子目录下，存储训练和测试的日志数据
            train_writer = tf.summary.FileWriter(self.log_dir + "/train", sess.graph)  # 这里必需将session的计算图加进去
            test_writer = tf.summary.FileWriter(self.log_dir + "/test")
            # 测试数据
            test_feed = {x: X_test, y_: y_test,keep_prob:1.0}
            for i in range(self.max_steps):  # 1000次训练
                if i % 100 == 0:  # 每100次进行测试
                    summary, acc = sess.run([merged, accuracy], feed_dict=test_feed)  # 数据汇总、测试精度
                    test_writer.add_summary(summary, i)  # 汇总结果、循环步数写入日志
                    log_train("Test accuracy at step %s,accuracy is: %g%%" % (i, acc * 100))
                else:  # 训练
                    x_train, y_train = mnist.train.next_batch(batch_size=self.batch_size) #100
                    if i % 100 == 50:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)  # 定义TF运行选项
                        run_metadata = tf.RunMetadata()  # 定义TF运行的元信息
                        summary, _ = sess.run([merged, train_step], feed_dict={x: x_train, y_: y_train,keep_prob:self.keep_prob},
                                              options=run_options, run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, "step%03d" % i)  # 将节点在运行时的元信息写入日志文件
                        train_writer.add_summary(summary, i)
                        # 注意，这里保存模型不是为了后期使用，而是为了可视化降维后的嵌入向量
                        saver.save(sess, self.log_dir + "/model.ckpt", i)
                        log_train(("Adding run metadata for"+str(i)))
                    else:
                        summary, _ = sess.run([merged, train_step], feed_dict={x: x_train, y_: y_train,keep_prob:self.keep_prob})
                        train_writer.add_summary(summary, i)
            # 关闭ＦileWriter
            train_writer.close()
            test_writer.close()

    def run1(self):
        def distinguish(X_run):
            digit = -1  # 最终识别的0-9的数字
            out,a_2_output = sess.run([y,hid_out], feed_dict={X: X_run,keep_prob:1.0})
            rst=sess.run(tf.nn.softmax(out))
            log_test('The output of the final model:{},{}'.format(rst,rst.shape)) #二维1x10
            max_prob=-0.1 #记录所有类别最大的概率值
            for idx in range(10):
                if max_prob  < rst[0][idx]:
                    max_prob = rst[0][idx]
                    digit = idx
            return rst,a_2_output,digit
        # 原测试单个样本1x784
        X_train, y_train, X_validation, y_validation, X_test, y_test, mnist = self.load_datasets()
        sample = X_test[102]#测试样本集的第103个样本
        img_in = sample.reshape(28, 28)  # ->28x28
        X_run = sample.reshape(1, 784)  # 从一维784变为二维1x784，作为模型输入样本
        # 自制样本
        img = io.imread(self.path3, as_gray=True)  # 以灰度图方式读取图像内容
        raw = [x for x in img.reshape(784)]  # 二维28x28->一维784，若每个像素<0.5置为1，否则0  (0 if x<0.4 else 1)
        make_sample = np.array(raw, dtype='float32')  # 变为numpy数组，作为一个样本
        make_img = make_sample.reshape(28, 28)  # ->28x28
        make_X = make_sample.reshape(1, 784)  # 从一维784变为二维1x784，作为模型输入样本
        # 加载图结构
        saver = tf.train.import_meta_graph("log/model.ckpt-950.meta")
        graph = tf.get_default_graph()  # 获取张量图
        X = graph.get_tensor_by_name('x-input:0')  # 获取输入变量（占位符）
        y_ = graph.get_tensor_by_name('y-input:0')  # 获取输入变量对应标签
        keep_prob = graph.get_tensor_by_name('keep_prob:0')  # 获取dropout的保留参数
        W_1 = graph.get_tensor_by_name('input_to_hidden/weights/Variable:0')
        hid_out = graph.get_tensor_by_name('input_to_hidden/activation:0')
        y = graph.get_tensor_by_name('hidden_to_out/activation:0')
        correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        # 恢复好模型后就可以进行测试了
        with tf.Session() as sess:
            # 获取最后一次训练好的模型参数,加载到当前环境中
            saver.restore(sess, tf.train.latest_checkpoint('log'))
            print('finish loading model!!!!!!!')
            # 全部测试样本精度
            test_accuracy= sess.run(accuracy, feed_dict={X: X_test, y_: y_test,keep_prob: 1.0})
            log_test('Test accuracy of all test samples of the final model:'+ str(test_accuracy))
            print('Identification of a single test sample:')
            # 原样本
            rst,a_2_output,digit=distinguish(X_run)
            wight_map=W_1[:,0].eval().reshape(28,28) #将第一列（所有连接到第一个隐层神经元的权值）变为28x28
            a_2_raw = a_2_output[0]#取出隐层输出的原始数据
            a_2_img = a_2_raw[0:484]#由于隐层输出512维数据，若显示成正方形数据，故只取前面484维
            feature_map = a_2_img.reshape(22,22)
            # 自制样本
            make_rst,make_a_2_output,make_digit=distinguish(make_X)
            make_a_2_raw = make_a_2_output[0]#取出隐层输出的原始数据
            make_a_2_img = make_a_2_raw[0:484]#由于隐层输出512维数据，若显示成正方形数据，故只取前面484维
            make_feature_map = make_a_2_img.reshape(22,22)
            plt.figure(1)
            plt.subplot(231);plt.imshow(img_in, cmap='gray')#输入图像
            plt.axis('off');plt.title('test_result:{0}'.format(digit))
            plt.subplot(232);plt.imshow(wight_map,cmap = 'gray')#输入层到隐层第一行连接权值图像
            plt.axis('off');plt.title('wight map')
            plt.subplot(233);plt.imshow(feature_map, cmap = 'gray')#隐层输出图像，特征图
            plt.axis('off');plt.title('feature map')
            plt.subplot(234);plt.imshow(make_img, cmap='gray')
            plt.axis('off');plt.title('make_result:{0}'.format(make_digit))
            plt.subplot(235);plt.imshow(wight_map,cmap = 'gray')
            plt.axis('off');plt.title('wight map')
            plt.subplot(236);plt.imshow(make_feature_map, cmap = 'gray')
            plt.axis('off');plt.title('feature map')
            plt.savefig('result/result1.jpg')
            plt.show()

    '''查看ckpt文件中保存的变量信息'''
    def check(self):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("log/model.ckpt-950.meta")  # 加载网络图
            graph = tf.get_default_graph()
            saver.restore(sess, tf.train.latest_checkpoint('log'))  # 获取最后一次训练好的模型参数
            reader = tf.train.NewCheckpointReader('log/model.ckpt-950') #模型名
            print("\n variable name, data type, shape:")
            print(reader.debug_string().decode("utf-8"))
            print(reader.get_tensor('hidden_to_out/biases/Variable')) #[10]
            # tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]  # 得到当前图中所有变量的名称
            for tensor in graph.as_graph_def().node:
                print(tensor.name)


if __name__=='__main__':
    start = time.clock()

    BP=my_BP()

    #训练
    BP.train()

    #查看变量
    # BP.check()

    #测试模型
    BP.run1()

    end = time.clock()
    print('运行时间：%f s'%float(end - start))
    pass


'''
activate tensorflow
tensorboard --logdir=E:/PycharmProjects/untitled2/DL/TB_my_BP/log/train  训练过程日志文件
http://LAPTOP-C8M2S6FN:6006

'''
'''打印的信息
Accuracy at step 0,accuracy is: 13.05%
Adding run metadata for 50
Accuracy at step 100,accuracy is: 91.59%
Adding run metadata for 150
Accuracy at step 200,accuracy is: 93.64%
Adding run metadata for 250
Accuracy at step 300,accuracy is: 94.38%
Adding run metadata for 350
Accuracy at step 400,accuracy is: 95.15%
Adding run metadata for 450
Accuracy at step 500,accuracy is: 95.27%
Adding run metadata for 550
Accuracy at step 600,accuracy is: 96.27%
Adding run metadata for 650
Accuracy at step 700,accuracy is: 96.34%
Adding run metadata for 750
Accuracy at step 800,accuracy is: 96.7%
Adding run metadata for 850
Accuracy at step 900,accuracy is: 96.8%
Adding run metadata for 950
运行时间：51.039034 s
'''
'''
变量名字, 数据类型, shape:
hidden_to_out/biases/Variable (DT_FLOAT) [10]
hidden_to_out/biases/Variable/Adam (DT_FLOAT) [10]
hidden_to_out/biases/Variable/Adam_1 (DT_FLOAT) [10]
hidden_to_out/weights/Variable (DT_FLOAT) [500,10]
hidden_to_out/weights/Variable/Adam (DT_FLOAT) [500,10]
hidden_to_out/weights/Variable/Adam_1 (DT_FLOAT) [500,10]
input_to_hidden/biases/Variable (DT_FLOAT) [500]
input_to_hidden/biases/Variable/Adam (DT_FLOAT) [500]
input_to_hidden/biases/Variable/Adam_1 (DT_FLOAT) [500]
input_to_hidden/weights/Variable (DT_FLOAT) [784,500]
input_to_hidden/weights/Variable/Adam (DT_FLOAT) [784,500]
input_to_hidden/weights/Variable/Adam_1 (DT_FLOAT) [784,500]
train/beta1_power (DT_FLOAT) []
train/beta2_power (DT_FLOAT) []
'''
'''
测试样本精度： 0.9695
单个测试样本的识别：
rst:[[1.3128135e-08 3.1705088e-08 3.9839744e-08 2.9522169e-04 4.5772424e-08
  9.9967372e-01 2.7789382e-09 2.4357475e-06 1.3985900e-05 1.4617132e-05]],(1, 10)
rst:[[2.7022962e-04 2.8091087e-04 1.2286340e-01 9.4742402e-03 1.4398397e-04
  4.6167749e-01 2.8609396e-03 7.1855044e-05 4.0173426e-01 6.2252203e-04]],(1, 10)
'''



