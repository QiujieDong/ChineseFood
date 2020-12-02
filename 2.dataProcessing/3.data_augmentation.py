from __future__ import print_function
from __future__ import division

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def augmentation(image_path, iters, file_list):

	image_datagen = ImageDataGenerator(rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		rescale=1. / 255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest',
		data_format='channels_last')
    
	for image_name in file_list:
		image = load_img(os.path.join(os.path.abspath(image_path), image_name))
        	
		try:
			# Image is converted to array type, because flow() receives numpy array as parameter 
			image_arr = img_to_array(image, data_format="channels_last")
			image_arr = image_arr.reshape((1,) + image_arr.shape)#flow() requires data to be 4 dimensions
		
			i = 0
			for _ in image_datagen.flow(
					image_arr, # this is the target directory
					batch_size=1,
					save_to_dir=image_path,
					save_prefix='aug', 
					save_format='jpg'):
				i += 1
				if i >= iters:
					break
		except:
			print('Skip image ', image_name)
			continue

if __name__ == '__main__':

    
	dataset_path = './ChineseFood/images/'
	filelist = os.listdir(dataset_path)
	for filename in filelist:
		if filename.startswith('.'):
			filelist.remove(file_name)
	filelist.sort()

	image_size = 2000 # about 2000 images every classify
	
	i = 0
	class_num = len(filelist)
	for filename in filelist:
		image_path = os.path.join(os.path.abspath(dataset_path), filename)

		file_list = os.listdir(image_path)
		for file_name in file_list:
			if file_name.startswith('.'):
				file_list.remove(file_name)

		image_num = len(file_list)
		iters = (image_size - image_num) // image_num # '//' has the same function as 'floor'
		
		print('Have done: {}/{}. Doing: {}.'.format(i, class_num, filename))
		augmentation(image_path, iters, file_list)
		i += 1
	print('All done!')
