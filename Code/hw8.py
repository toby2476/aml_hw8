import numpy as np
import pandas as pd
import mnist
import scipy.misc
from PIL import Image

def import_data():

	train_images = mnist.train_images()
	train_images = train_images.reshape((train_images.shape[0], train_images.shape[1]*train_images.shape[2]))
	train_labels = mnist.train_labels()
	update_coords = pd.read_csv('../Data/SupplementaryAndSampleData/UpdateOrderCoordinates.csv',sep=',')
	initial_params = pd.read_csv('../Data/SupplementaryAndSampleData/InitialParametersModel.csv',sep=',',header=None)
	initial_params = np.array(initial_params)
	return train_images[0:20,:],update_coords,initial_params

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


def save_images(images,description):
	for i in range(len(images)):
		image_out = np.zeros((images.shape[0],images.shape[1]))
		image_out[images==-1] = 0
		image_out[images==1] = 255
		image_out = image_out.reshape(image_out.shape[0],28,28)
		image_out = image_out.astype('uint8')
		im = Image.fromarray(image_out[i,:,:],mode='L')
		im.save('../Output/Image_'+str(i)+'_'+description+'.png')
		
def save_predictions(predictions):
	for i in range(len(predictions)):
		np.savetxt('../Output/Image_'+str(i)+'_Prediction',predictions[i],delimiter='\t',fmt='%d')

def mean_field_inference(update_coords,train_images,pi,theta_xh,theta_hh,num_iter,img_num):
	
	for a in range(num_iter):		
		for i in range(len(train_images[img_num,:])):
			row = update_coords['Pixel '+str(i)].iloc[2*img_num]
			col = update_coords['Pixel '+str(i)].iloc[2*img_num + 1]
			neighbors = []
			if(row != 0): neighbors.append(pi[row-1,col])
			if(row != len(pi)-1): neighbors.append(pi[row+1,col])
			if(col != 0): neighbors.append(pi[row,col-1])
			if(col != len(pi[0,:])-1): neighbors.append(pi[row,col+1])
			sum1 = 0
			sum3 = 0
			for j in neighbors:
				sum1 += theta_hh*(2*j - 1)
				sum3 += -theta_hh*(2*j - 1)
			sum2 = theta_xh * train_images[img_num,row*len(pi[0,:])+col]
			sum4 = -theta_xh * train_images[img_num,row*len(pi[0,:])+col]
			pi_i = (np.exp(sum1 + sum2))/(np.exp(sum1 + sum2)+np.exp(sum3 + sum4))
			pi[row,col] = pi_i

	return pi

def reconstruct_image(pi):
	image = np.zeros(pi.shape)
	image[pi<0.5] = -1
	image[pi>=0.5] = 1
	return image
	
def mean_field_wrapper(update_coords,train_images,initial_params,theta_xh,theta_hh,num_iter):
	denoised_images = np.zeros(train_images.shape)
	predictions = []
	for i in range(len(train_images)):
		pi = mean_field_inference(update_coords,train_images,initial_params,theta_xh,theta_hh,num_iter,i)
		image = reconstruct_image(pi)
		image = image.flatten()
		denoised_images[i,:] = image
		pred = pi
		pred[pred<0.5] = 0
		pred[pred>=0.5] = 1
		predictions.append(pred)
		

	return denoised_images,predictions
	


def main():

	theta_xh = 2
	theta_hh = 0.8
	num_iter = 10
	train_images,update_coords,initial_params = import_data()
	train_images = binarize(train_images)
	save_images(train_images,'original')
	train_images = add_noise(train_images)
	save_images(train_images,'noisy')
	denoised_images, predictions = mean_field_wrapper(update_coords,train_images,initial_params,theta_xh,theta_hh,num_iter)
	save_images(denoised_images,'denoised')
	save_predictions(predictions)

main()
