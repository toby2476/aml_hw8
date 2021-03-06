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

def add_noise(images):
	noisy_images = np.copy(images)
	noise_coords = pd.read_csv('../Data/SupplementaryAndSampleData/NoiseCoordinates.csv',sep=',')
	for i in range(len(noisy_images)):
		for j in range(15):		
			image_row = noise_coords['Noisy bit '+str(j)].iloc[i*2]
			image_col = noise_coords['Noisy bit '+str(j)].iloc[i*2 + 1]
			bit_idx = image_row*28 + image_col
			noisy_images[i,bit_idx] *= -1 	

	return noisy_images


def save_images(orig_images,noisy_images,denoised_images):
	saved_image = np.zeros((84,280))
	images = [orig_images,noisy_images,denoised_images]
	for i in range(len(images)):
		for j in range(10,len(orig_images)):
			image_out = np.zeros(images[i][j,:].shape)
			image_out[images[i][j,:]==-1] = 0
			image_out[images[i][j,:]==1] = 255
			image_out = image_out.reshape((28,28))
			saved_image[28*i:28*(i+1),28*(j-10):28*(j-9)] = image_out

	saved_image = saved_image.astype('uint8')
	im = Image.fromarray(saved_image,mode='L')
	im.save('../Output/Images.png')
		
def save_files(predictions,energy_mat):
	pred = np.zeros((28,280))
	for i in range(10,len(predictions)):
		pred[:,28*(i-10):28*(i-9)] = predictions[i]	
	np.savetxt('../Output/denoised.csv',pred,delimiter=',',fmt='%d')
	np.savetxt('../Output/energy.csv',energy_mat[10:12,0:2],delimiter=',',fmt='%f')
	np.savetxt('../Output/energy_all_values.csv',energy_mat,delimiter=',',fmt='%f')

def mean_field_inference(update_coords,train_images,params,theta_xh,theta_hh,num_iter,img_num):
	
	energy_list = []
	pi = np.copy(params)
	init_energy = energy_function(pi,theta_xh,theta_hh,train_images,img_num)
	energy_list.append(init_energy)
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
		
		energy = energy_function(pi,theta_xh,theta_hh,train_images,img_num)
		energy_list.append(energy)		

	return pi, energy_list

def reconstruct_image(pi):
	image = np.zeros(pi.shape)
	image[pi<0.5] = -1
	image[pi>=0.5] = 1
	return image
	
def energy_function(pi,theta_xh,theta_hh,images,img_num):
	term1 = 0
	term2 = 0
	for i in range(pi.shape[0]):
		for j in range(pi.shape[1]):
			term1 += pi[i,j]*np.log(pi[i,j]+10e-10) + (1-pi[i,j])*np.log(1-pi[i,j]+10e-10)
			neighbors = []
			if(i != 0): neighbors.append(pi[i-1,j])
			if(i != len(pi)-1): neighbors.append(pi[i+1,j])
			if(j != 0): neighbors.append(pi[i,j-1])
			if(j != len(pi[0,:])-1): neighbors.append(pi[i,j+1])
			for k in neighbors:
				term2 += theta_hh*(2*pi[i,j]-1)*(2*k - 1)
			term2 += theta_xh*(2*pi[i,j]-1)*images[img_num,i*pi.shape[1] + j]
	
	energy = term1 - term2
	return energy	
				

def mean_field_wrapper(update_coords,train_images,initial_params,theta_xh,theta_hh,num_iter):
	denoised_images = np.zeros(train_images.shape)
	predictions = []
	energy_mat = np.zeros((train_images.shape[0],num_iter+1))
	for i in range(len(train_images)):
		pi, energy = mean_field_inference(update_coords,train_images,initial_params,theta_xh,theta_hh,num_iter,i)
		image = reconstruct_image(pi)
		image = image.flatten()
		denoised_images[i,:] = image
		pred = np.copy(pi)
		pred[pred<0.5] = 0
		pred[pred>=0.5] = 1
		predictions.append(pred)
		energy_mat[i,:] = energy
		
	return denoised_images,predictions,energy_mat
	


def main():

	theta_xh = 2
	theta_hh = 0.8
	num_iter = 10
	orig_images,update_coords,initial_params = import_data()
	orig_images = binarize(orig_images)
	noisy_images = add_noise(orig_images)
	denoised_images, predictions, energy_mat = mean_field_wrapper(update_coords,noisy_images,initial_params,theta_xh,theta_hh,num_iter)
	save_images(orig_images,noisy_images,denoised_images)
	save_files(predictions,energy_mat)

main()
