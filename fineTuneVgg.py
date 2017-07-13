import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from vgg19 import Vgg19
from datagenerator import ImageDataGenerator
import cv2

# Path to the textfiles for the trainings and validation set
train_file = 'train.txt'
val_file = 'validation.txt'

# Learning params
learning_rate = 0.01
num_epochs = 100
batch_size = 1

# Network params
dropout_rate = 0.5
num_classes = 17
skipLayer=['fc8', 'fc7']
train_layers = ['fc8', 'fc7','fc6','conv5_4','conv5_3','conv5_2','conv5_1','conv4_4','conv4_3','conv4_2','conv4_1']

# Path for tf.summary.FileWriter and to store model checkpoints
checkpoint_path = "tmpCnn/"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = Vgg19(x, keep_prob, num_classes, skipLayer)

# Link variable to model output
score = model.fc8
softmax_score=tf.nn.softmax(score)
# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))  

# Train op
with tf.name_scope("train"):
	# Get gradients of all trainable variables
	gradients = tf.gradients(loss, var_list)
	gradients = list(zip(gradients, var_list))
	# Create optimizer and apply gradient descent to the trainable variables
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.apply_gradients(grads_and_vars=gradients)
# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
	correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file, horizontal_flip = True, shuffle = True)
val_generator = ImageDataGenerator(val_file, shuffle = False,testData=False) 
val_batches_per_epoch=np.ceil(val_generator.data_size / float(batch_size)).astype(np.int32)
train_batches_per_epoch = np.ceil(train_generator.data_size / float(batch_size)).astype(np.int32)


# Start Tensorflow session

def trainModel():
	labelList=np.zeros([val_generator.data_size,17])
	acList=[]
	with open("valLabels.txt","r") as f:
		i=0
		for line in f:
			labels=map(int,line.strip().split())
			for l in labels:
				labelList[i][l]=1
			i+=1
	with tf.Session() as sess:
		restore=False
		if restore:
			saver_restore= tf.train.Saver()
			saver_restore.restore(sess,tf.train.latest_checkpoint(checkpoint_path))
			losses = 0.
			start=0
			for _ in range(val_batches_per_epoch):
				batch_tx, batch_ty = val_generator.next_batch(batch_size)
				re = sess.run(softmax_score, feed_dict={x: batch_tx,keep_prob: 1.})
				length=len(batch_tx)
				batchLabel=labelList[start:start+length]
				start+=length
				for j in range(len(batchLabel)):
					for m in range(len(batchLabel[j])):
						losses+=pow(float(batchLabel[j][m])-float(re[j][m]),2)
			losses=losses/val_generator.data_size
			val_generator.reset_pointer()
			acList.append(losses)
		else:
			sess.run(tf.global_variables_initializer())
			model.loadModel(sess)
			acList.append(10)
		print("{} Start training...".format(datetime.now()))
		# Loop over number of epochs
		for epoch in range(0,num_epochs):
			print("{} Epoch number: {}".format(datetime.now(), epoch+1))
			  
			step = 1
			  
			while step < train_batches_per_epoch:			  
				# Get a batch of images and labels
				batch_xs, batch_ys = train_generator.next_batch(batch_size)				  
				# And run the training op
				sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_rate})	  
				step += 1
				print step
			  
			  #Validate the model on the entire validation set
			print("{} Start validation".format(datetime.now()))
			losses = 0.
			start=0
			for _ in range(val_batches_per_epoch):
				batch_tx, batch_ty = val_generator.next_batch(batch_size)
				re = sess.run(softmax_score, feed_dict={x: batch_tx,keep_prob: 1.})
				length=len(batch_tx)
				batchLabel=labelList[start:start+length]
				start+=length
				for j in range(len(batchLabel)):
						for m in range(len(batchLabel[j])):
							losses+=pow(float(batchLabel[j][m])-float(re[j][m]),2)
			losses=losses/val_generator.data_size
			print("loss of"+str(epoch+1)+":",losses)
			if np.min(acList)>losses:
				acList.append(losses)
				print("{} Saving checkpoint of model...".format(datetime.now()))  
				#save checkpoint of the model
				checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
				save_path = saver.save(sess, checkpoint_name)  
				print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
			val_generator.reset_pointer()
			train_generator.reset_pointer()

	
###generate testResult
def generateTestResult():
	size=val_generator.data_size
	size_i=0
	sess=tf.Session()
	checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(24)+'.ckpt')
	saver.restore(sess,checkpoint_name)
	write=open("result.txt","a+")
	threshold=0.15
	idLabel={}
	it=0
	with open("label.txt","r") as f:
		for line in f:
			content=line.strip().split()
			id=int(content[1])
			name=content[0]
			idLabel[id]=name
	for _ in range(val_batches_per_epoch):
		it+=1
		print it
		batch_tx= val_generator.next_batch(batch_size)
		result=sess.run(softmax_sore,feed_dict={x:batch_tx, keep_prob: 1.0})
		for i in range(len(result)):
			if size_i>=size:
				break
			size_i+=1
			d={}
			for j in range(len(result[i])):
				d[j]=result[i][j]
			d=sorted(d.iteritems(), key=lambda d:d[1], reverse = True)
			keyMax,valMax=d[0]
			write.write(str(idLabel[int(keyMax)]))
			for key,val in d:
				if valMax-val<threshold and valMax-val!=0:
					write.write("\t"+str(idLabel[int(key)]))
			write.write("\n")
			print ("done a batch")
	write.close()

###generate testfeature
def generFeatureTest():
	size=val_generator.data_size
	size_i=0
	sess=tf.Session()
	checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(24)+'.ckpt')
	saver.restore(sess,checkpoint_name)
	write=open("featureTest.txt","a+")
	idLabel={}
	with open("label.txt","r") as f:
		for line in f:
			content=line.strip().split()
			id=int(content[1])
			name=content[0]
			idLabel[id]=name
	for _ in range(val_batches_per_epoch):
		batch_tx= val_generator.next_batch(batch_size)
		result=sess.run(softmax_score,feed_dict={x:batch_tx, keep_prob: 1.0})
		for i in range(len(result)):
			if size_i>=size:
				break
			size_i+=1
			d={}
			for j in range(len(result[i])):
				d[j]=result[i][j]
			d=sorted(d.iteritems(), key=lambda d:d[1], reverse = True)
			for key,val in d:
				write.write(str(idLabel[int(key)])+":"+str(val)+" ")
			write.write("\n")
			print ("done a batch")
	write.close()

def generFeatureVal():
	labelList=np.zeros([val_generator.data_size,17])
	size=val_generator.data_size
	size_i=0
	sess=tf.Session()
	checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(24)+'.ckpt')
	saver.restore(sess,checkpoint_name)
	write=open("featureVal.txt","a+")
	idLabel={}
	with open("valLabels.txt","r") as f:
		i=0
		for line in f:
			labels=map(int,line.strip().split())
			for l in labels:
				labelList[i][l]=1
			i+=1
	with open("label.txt","r") as f:
		for line in f:
			content=line.strip().split()
			id=int(content[1])
			name=content[0]
			idLabel[id]=name
	start=0
	for _ in range(val_batches_per_epoch):
		batch_tx= val_generator.next_batch(batch_size)
		result=sess.run(softmax_score,feed_dict={x:batch_tx, keep_prob: 1.0})
		length=len(batch_tx)
		if start+length<=size:
			batchLabel=labelList[start:start+length]
		else:
			batchLabel=labelList[start:]
		start+=length
		for i in range(len(result)):
			if size_i>=size:
				break
			size_i+=1
			d={}
			for j in range(len(result[i])):
				d[j]=result[i][j]
			d=sorted(d.iteritems(), key=lambda d:d[1], reverse = True)
			for key,val in d:
				write.write(str(idLabel[int(key)])+":"+str(val)+" ")
			write.write(",")
			for j in range(17):
				if int(batchLabel[i][j])==1:
					write.write(str(idLabel[int(j)])+" ")
			write.write("\n")
		print ("done a batch")
	write.close()

def generFeaturePool5():
	size=val_generator.data_size
	size_i=0
	sess=tf.Session()
	checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(31)+'.ckpt')
	saver.restore(sess,checkpoint_name)
	write=open("featurePool5.txt","a+")
	for _ in range(val_batches_per_epoch):
		batch_tx= val_generator.next_batch(batch_size)
		result=sess.run(pool5,feed_dict={x:batch_tx, keep_prob: 1.0})
		for i in range(len(result)):
			if size_i>=size:
				break
			size_i+=1
			for j in range(len(result[i])):
				write.write(str(result[i][j])+" ")
			write.write("\n")
		print ("done a batch")
	write.close()

##main
if __name__=="__main__":
	# generateFeature()
	trainModel()
	# generateTestResult()
	# generFeaturePool5()