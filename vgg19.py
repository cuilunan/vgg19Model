import tensorflow as tf
import numpy as np


class Vgg19(object):
	def __init__(self,x,keep_prob,numClasses,skipLayer,weightPath="vgg19.npy"):
		self.X=x
		self.NUM_CLASSES=numClasses
		self.KEEP_PROB=keep_prob
		self.SKIP_LAYER=skipLayer
		self.WEIGHT_PATH=weightPath
		self.create()
	def create(self):
		self.conv1_1 = self.convLayer(self.X, 3, 3, 1, 1, 64, "conv1_1" )
		self.conv1_2 = self.convLayer(self.conv1_1, 3, 3, 1, 1, 64, "conv1_2")
		self.pool1 = self.maxPoolLayer(self.conv1_2, 2, 2, 2, 2, "pool1")

		self.conv2_1 = self.convLayer(self.pool1, 3, 3, 1, 1, 128, "conv2_1")
		self.conv2_2 = self.convLayer(self.conv2_1, 3, 3, 1, 1, 128, "conv2_2")
		self.pool2 = self.maxPoolLayer(self.conv2_2, 2, 2, 2, 2, "pool2")

		self.conv3_1 = self.convLayer(self.pool2, 3, 3, 1, 1, 256, "conv3_1")
		self.conv3_2 = self.convLayer(self.conv3_1, 3, 3, 1, 1, 256, "conv3_2")
		self.conv3_3 = self.convLayer(self.conv3_2, 3, 3, 1, 1, 256, "conv3_3")
		self.conv3_4 = self.convLayer(self.conv3_3, 3, 3, 1, 1, 256, "conv3_4")
		self.pool3 = self.maxPoolLayer(self.conv3_4, 2, 2, 2, 2, "pool3")

		self.conv4_1 = self.convLayer(self.pool3, 3, 3, 1, 1, 512, "conv4_1")
		self.conv4_2 = self.convLayer(self.conv4_1, 3, 3, 1, 1, 512, "conv4_2")
		self.conv4_3 = self.convLayer(self.conv4_2, 3, 3, 1, 1, 512, "conv4_3")
		self.conv4_4 = self.convLayer(self.conv4_3, 3, 3, 1, 1, 512, "conv4_4")
		self.pool4 = self.maxPoolLayer(self.conv4_4, 2, 2, 2, 2, "pool4")

		self.conv5_1 = self.convLayer(self.pool4, 3, 3, 1, 1, 512, "conv5_1")
		self.conv5_2 = self.convLayer(self.conv5_1, 3, 3, 1, 1, 512, "conv5_2")
		self.conv5_3 = self.convLayer(self.conv5_2, 3, 3, 1, 1, 512, "conv5_3")
		self.conv5_4 = self.convLayer(self.conv5_3, 3, 3, 1, 1, 512, "conv5_4")
		self.pool5 = self.maxPoolLayer(self.conv5_4, 2, 2, 2, 2, "pool5")

		self.fcIn = tf.reshape(self.pool5, [-1, 7*7*512])
		self.fc6 = self.fcLayer(self.fcIn, 7*7*512, 4096, True, "fc6")
		self.dropout1 = self.dropout(self.fc6, self.KEEP_PROB)

		self.fc7 = self.fcLayer(self.dropout1, 4096, 4096, True, "fc7")
		self.dropout2 = self.dropout(self.fc7, self.KEEP_PROB)

		self.fc8 = self.fcLayer(self.dropout2, 4096, self.NUM_CLASSES, True, "fc8")

	def convLayer(self,x, kHeight, kWidth, strideX, strideY,
				  featureNum, name, padding = "SAME"):
		channel = int(x.get_shape()[-1])
		with tf.variable_scope(name) as scope:
			w = tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum])
			b = tf.get_variable("b", shape = [featureNum])
			featureMap = tf.nn.conv2d(x, w, strides = [1, strideY, strideX, 1], padding = padding)
			out = tf.nn.bias_add(featureMap, b)
			# return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name = scope.name)
			return tf.nn.relu(out, name = scope.name)

	def maxPoolLayer(self,x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
		return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
						  strides = [1, strideX, strideY, 1], padding = padding, name = name)

	def fcLayer(self,x, inputD, outputD, reluFlag, name):
		with tf.variable_scope(name) as scope:
			w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float",initializer=tf.contrib.layers.xavier_initializer())
			b = tf.get_variable("b", [outputD], dtype = "float",initializer=tf.constant_initializer(0.1))
			out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
			if reluFlag:
				return tf.nn.relu(out)
			else:
				return out

	def dropout(self,x, keepPro, name = None):
		return tf.nn.dropout(x, keepPro, name)

	def loadModel(self, sess):
		wDict = np.load(self.WEIGHT_PATH, encoding = "bytes").item()
		#for layers in model
		for name in wDict:
			if name not in self.SKIP_LAYER:
				with tf.variable_scope(name, reuse = True):
					for p in wDict[name]:
						if len(p.shape) == 1:
							#bias
							sess.run(tf.get_variable('b', trainable = False).assign(p))
						else:
							#weights
							sess.run(tf.get_variable('w', trainable = False).assign(p))