import numpy as np
import cv2
import copy

class ImageDataGenerator:
	def __init__(self, class_list, horizontal_flip=False, shuffle=False, 
				 mean = np.array([104., 117., 124.]), scale_size=(224, 224),
				 nb_classes = 17,testData=False):
		# Init params
		self.horizontal_flip = horizontal_flip
		self.n_classes = nb_classes
		self.shuffle = shuffle
		self.mean = mean
		self.scale_size = scale_size
		self.pointer = 0
		self.TEST_DATA=testData
		
		self.read_class_list(class_list)
		
		if self.shuffle:
			self.shuffle_data()

	def read_class_list(self,class_list):
		"""
		Scan the image file and get the image paths and labels
		"""
		if self.TEST_DATA!=True:
			with open(class_list) as f:
				lines = f.readlines()
				self.images = []
				self.labels = []
				for l in lines:
					items = l.split()
					self.images.append(items[0])
					self.labels.append(int(items[1]))
				
				#store total number of data
				self.data_size = len(self.labels)
		else:
			with open(class_list) as f:
				lines = f.readlines()
				self.images = []
				for l in lines:
					items = l.split()
					self.images.append(items[0])
				
				#store total number of data
				self.data_size = len(self.images)

	def shuffle_data(self):
		"""
		Random shuffle the images and labels
		"""
		images = copy.deepcopy(self.images)
		labels = copy.deepcopy(self.labels)
		self.images = []
		self.labels = []
		
		#create list of permutated index and shuffle data accoding to list
		idx = np.random.permutation(len(labels))
		for i in idx:
			self.images.append(images[i])
			self.labels.append(labels[i])
				
	def reset_pointer(self):
		"""
		reset pointer to begin of the list
		"""
		self.pointer = 0
		
		if self.shuffle:
			self.shuffle_data()
		
	
	def next_batch(self, batch_size):
		if self.TEST_DATA!=True:
			paths = self.images[self.pointer:self.pointer + batch_size]
			labels = self.labels[self.pointer:self.pointer + batch_size]
			
			#update pointer
			self.pointer += batch_size
			
			# Read images
			images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
			for i in range(len(paths)):
				img = cv2.imread(paths[i])

				#flip image at random if flag is selected
				if self.horizontal_flip and np.random.random() < 0.5:
					img = cv2.flip(img, 1)
				
				#rescale image
				img = cv2.resize(img, (self.scale_size[0], self.scale_size[0]))

				img = img.astype(np.float32)
				
				#subtract mean
				img -= self.mean
																	 
				images[i] = img

			# Expand labels to one hot encoding
			one_hot_labels = np.zeros((batch_size, self.n_classes))
			for i in range(len(labels)):
				one_hot_labels[i][labels[i]] = 1

			#return array of images and labels
			return images, one_hot_labels
		else:
			if self.pointer+batch_size>len(self.images):
				paths=self.images[self.pointer:]
				num=batch_size-len(paths)
				for i in range(num):
					paths.append(self.images[0])
			else:
				paths = self.images[self.pointer:self.pointer + batch_size]
			
			#update pointer
			self.pointer += batch_size
			
			# Read images
			images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
			for i in range(len(paths)):
				img = cv2.imread(paths[i])
				#flip image at random if flag is selected
				if self.horizontal_flip and np.random.random() < 0.5:
					img = cv2.flip(img, 1)
				
				#rescale image
				img = cv2.resize(img, (self.scale_size[0], self.scale_size[0]))
				img = img.astype(np.float32)
				
				#subtract mean
				img -= self.mean
																	 
				images[i] = img

			#return array of images and labels
			return images
