# 训练权重复用
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names
from VGG16_model import Vgg16
n_class = 1000




if "__main__" == __name__:

    # 定义输入图像占位符
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])

    vgg = Vgg16(imgs)
    prob = vgg.probs

    # =====模型存储
    saver = vgg.saver()
    with tf.Session() as sess:

        # 加载训练好的vgg16权重文件
        vgg.load_weight("./vgg16_weights.npz", sess)
        saver.save(sess, "./model/vgg.ckpt")

    # =====模型恢复
    saver = vgg.saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model/vgg.ckpt")

        # 加载图像文件
        img1 = imread("001.jpg", mode="RGB")
        img1 = imresize(img1, (224, 224))

        # 预测
        prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]

        # 生成一系列key值和对应的概率值，选择最有可能的前5个
        preds = (np.argsort(prob)[::-1])[0:5]

        for p in preds:
            # 输出类名称及对应概率
            print(class_names[p], prob[p])