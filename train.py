# import tensorflow as tf
import tensorflow.compat.v1 as tf
import scipy.io as sio
import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_model():
	learning_rate = 0.001
	# inputs_ = tf.placeholder(tf.float32, (None, 200, 200, 3), name='inputs')
	# targets_ = tf.placeholder(tf.float32, (None, 200, 200, 3), name='targets')


	# ### Encoder
	# conv1 = tf.layers.conv2d(inputs=inputs_, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# # Now 200x200x32
	# maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
	# # Now 100x100x32
	# conv2 = tf.layers.conv2d(inputs=maxpool1, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# # Now 100x100x64
	# maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
	# # Now 50x50x64
	# conv3 = tf.layers.conv2d(inputs=maxpool2, filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# # Now 50x50x128
	# encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
	# # Now 25x25x128

	# bottle_neck = tf.layers.conv2d(inputs=encoded, filters=256, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# # Now 25x25x256

	# ### Decoder
	# upsample1 = tf.image.resize_images(bottle_neck, size=(50,50), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	# # Now 50x50x256
	# conv4 = tf.layers.conv2d(inputs=upsample1, filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# # Now 50x50x128
	# upsample2 = tf.image.resize_images(conv4, size=(100,100), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	# # Now 100x100x128
	# conv5 = tf.layers.conv2d(inputs=upsample2, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# # Now 100x100x64
	# upsample3 = tf.image.resize_images(conv5, size=(200,200), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	# # Now 200x200x64
	# conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# # Now 200x200x32
	# logits = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=(3,3), padding='same', activation=None)
	# #Now 200x200x3

	# ----------------------------------------------------------------------------
	inputs_ = tf.placeholder(tf.float32, (None, 128, 128, 3), name='inputs')
	targets_ = tf.placeholder(tf.float32, (None, 128, 128, 1), name='targets')

	### Encoder
	conv1 = tf.layers.conv2d(inputs=inputs_, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 200x200x32
	maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
	# Now 100x100x32
	conv2 = tf.layers.conv2d(inputs=maxpool1, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 100x100x64
	maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
	# Now 50x50x64
	conv3 = tf.layers.conv2d(inputs=maxpool2, filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 50x50x128
	encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
	# Now 25x25x128

	bottle_neck = tf.layers.conv2d(inputs=encoded, filters=256, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 25x25x256

	### Decoder
	upsample1 = tf.image.resize_images(bottle_neck, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	# Now 50x50x256
	conv4 = tf.layers.conv2d(inputs=upsample1, filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 50x50x128
	upsample2 = tf.image.resize_images(conv4, size=(64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	# Now 100x100x128
	conv5 = tf.layers.conv2d(inputs=upsample2, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 100x100x64
	upsample3 = tf.image.resize_images(conv5, size=(128,128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	# Now 200x200x64
	conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	# Now 200x200x32
	logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3), padding='same', activation=None)
	#Now 200x200x3
	# -------------------------------------------------------------------------------------
	
	# Pass logits through sigmoid to get reconstructed image
	decoded = tf.nn.sigmoid(logits)
	# Pass logits through sigmoid and calculate the cross-entropy loss
	loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
	# Get cost and define the optimizer
	cost = tf.reduce_mean(loss)
	opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	# opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	model = {'inputs':inputs_, 'targets':targets_, 'pred': tf.sigmoid(logits), 'cost':cost, 'opt':opt }
	return model

def get_data():
	num_data = 500
	# energy = []
	# phase = []
	# crack = []
	# for i in range(1,num_data,1):
	# 	energy += [cv2.imread('./images/energy_{}.png'.format(i))]
	# 	phase +=  [cv2.imread('./images/phase_{}.png'.format(i))]
	# 	crack +=  [cv2.imread('./images/crack_{}.png'.format(i))]
	# energy = np.stack(energy,0).astype('float32')/255.
	# phase = np.stack(phase,0).astype('float32')/255.
	# crack = np.stack(crack,0).astype('float32')/255.

	#-------------------------------------
	for samples in range(num_data):
		data = sio.loadmat('./Arrays/Sample'+str(samples+1)+'.mat')

		# load data shape is height, width, timesteps
		crackk = data['Crack'].transpose() # shape: timesteps, height, width
		microstruc = crackk[:1, ...]
		crackk = crackk - np.repeat(microstruc, crackk.shape[0], axis=0)

		crackk = np.take(crackk, [0, 9, 19, 29, 39, 49], axis=0) #extract crack at timesteps 10, 20, 30, 40, 50
		
		Ux = data['Ux'].transpose()[1, ...] # shape: height, width
		Uy = data['Uy'].transpose()[1, ...]

		# in case rigid body motion in LPM model
		Ux = np.where(Ux > 0.01, 0.01, Ux)
		Ux = np.where(Ux < -0.01, -0.01, Ux)
		Uy = np.where(Uy > 0.01, 0.01, Uy)
		Uy = np.where(Uy < -0.01, -0.01, Uy)


		# add aditional dimision of channels
		Ux = np.expand_dims(Ux, 0)
		Uy = np.expand_dims(Uy, 0)
		displacement = np.concatenate((Ux, Uy), axis=0)

		# add aditional dimision of samples
		microstruc = np.expand_dims(microstruc, 0)
		displacement = np.expand_dims(displacement, 0)
		crackk = np.expand_dims(crackk, 0)

		input = np.concatenate((displacement, microstruc), axis=1) #shape: samples, channels, height, width
		# input = microstruc
		truth = crackk[:, 4:5, ...]

		if samples == 0:
			phase = input
			crack = truth
		else:
			phase = np.concatenate((phase, input), axis=0)
			crack = np.concatenate((crack, truth), axis=0)

	phase = np.transpose(phase, axes=[0, 2, 3, 1]) # shape: samples, height, width, channels
	crack = np.transpose(crack, axes=[0, 2, 3, 1])
	# ----------------------------------------------------------------------


	train_input = phase[:450, ...]
	train_output = crack[:450, ...]
	test_input = phase[450:, ...]
	test_output = crack[450:, ...]
	print(train_input.shape, train_output.shape, test_input.shape, test_output.shape)
	data = {'train_input':train_input, 'train_output':train_output, 'test_input':test_input, 'test_output':test_output}

	return data

def train_model(model, data):
	sess = tf.Session()
	epochs = 400
	# Set's how much noise we're adding to the MNIST images
	sess.run(tf.global_variables_initializer())
	cost_hist = {'train': [], 'test': []}
	for e in range(epochs):
		train_batch_cost = []
		for i in range(45):
			feed_dict = {model['inputs']: data['train_input'][10*i:10*(i+1)], model['targets']: data['train_output'][10*i:10*(i+1)]}
			train_batch_cost_i, _ = sess.run([model['cost'], model['opt']], feed_dict=feed_dict)
			train_batch_cost += [train_batch_cost_i]
		train_batch_cost = np.mean(train_batch_cost)
		test_batch_cost, pred = sess.run([model['cost'], model['pred']], feed_dict={model['inputs']: data['test_input'], model['targets']: data['test_output']})
		print("Epoch: {}/{}...".format(e+1, epochs), "Training loss: {:.4f}".format(train_batch_cost), "Testing loss: {:.4f}".format(test_batch_cost))

		# -------------post process predicted results------------
		pred = np.where(pred<0.4, 0, pred)
		pred = np.where(pred>=0.4, 1, pred)
		# -------------------------------------------------------

		cost_hist['train'] += [train_batch_cost]
		cost_hist['test'] += [test_batch_cost]
		np.save('loss_hist',cost_hist)
		np.save('pred',pred)
		sio.savemat('test_data.mat', {'input': data['test_input'], 'output': data['test_output'], 'pred': pred})

		plt.figure(figsize=(8,2))
		for i in range(10):
			plt.subplot(3,10,i+1)
			# plt.imshow(data['test_input'][i])
			plt.imshow(data['test_input'][i, :, :, -1])
			plt.axis('off')
			plt.subplot(3,10,i+10+1)
			# plt.imshow(data['test_output'][i])
			plt.imshow(np.squeeze(data['test_output'][i, ...]))
			plt.axis('off')
			plt.subplot(3,10,i+20+1)
			# plt.imshow(pred[i])
			plt.imshow(np.squeeze(pred[i, ...]))
			plt.axis('off')
		plt.savefig('ep_{}'.format(e), dpi=300)
		plt.close()
	
if __name__ == '__main__':
	tf.compat.v1.disable_eager_execution()
	data = get_data()
	model = create_model()
	train_model(model,data)


