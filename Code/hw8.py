import numpy as np
import pandas as pd
import mnist
import scipy.misc
from PIL import Image

def import_data():

	train_images = mnist.train_images()
	train_images = train_images.reshape((train_images.shape[0], train_images.shape[1]*train_images.shape[2]))
	train_labels = mnist.train_labels()
	
	return train_images[0:19,:],train_labels[0:19]

def binarize(train_images):
	train_images[train_images<128] = -1
	train_images[train_images>=128] = 1
	return train_images

def add_noise(train_images):

	noise_coords = pd.read_csv('../Data/SupplementaryAndSampleData/NoiseCoordinates.csv',sep=',')
	for i in range(len(train_images)):
		for j in range(15):		
			image_row = noise_coords['Noisy bit '+str(j)].iloc[i*2]
			image_col = noise_coords['Noisy bit '+str(j)].iloc[i*2 + 1]
			bit_idx = image_row*28 + image_col
			train_images[i,bit_idx] *= -1 	

	return train_images


def display_image(train_images,i):
	image_out = np.zeros((train_images.shape[0],train_images.shape[1]))
	image_out[train_images==-1] = 0
	image_out[train_images==1] = 255
	image_out = image_out.reshape(image_out.shape[0],28,28)
	im = Image.fromarray(image_out[i,:,:])
	im.show()

def display_all(train_images):
	for i in range(len(train_images)):
		display_image(train_images,i)

def main():

	train_images, train_labels = import_data()
	train_images = binarize(train_images)
	display_image(train_images,4)
	train_images = add_noise(train_images)
	display_image(train_images,4)

main()
