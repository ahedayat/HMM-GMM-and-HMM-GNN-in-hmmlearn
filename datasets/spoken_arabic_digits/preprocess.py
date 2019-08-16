import os
import shutil
import time
import xml.etree.ElementTree as ET

def mkdir(dir_path, dir_name, forced_remove=False):
	new_dir = '{}/{}'.format(dir_path,dir_name)
	if forced_remove and os.path.isdir( new_dir ):
		shutil.rmtree( new_dir )
	if not os.path.isdir( new_dir ):
		os.makedirs( new_dir )
def touch(file_path, file_name, forced_remove=False):
	new_file = '{}/{}'.format(file_path,file_name)
	assert os.path.isdir( file_path ), ' \"{}\" does not exist.'.format(file_path)
	if forced_remove and os.path.isfile(new_file):
		os.remove(new_file)
	if not os.path.isfile(new_file):
		open(new_file, 'a').close()

def write_file(file_path, file_name, content, forced_remove_prev=True, new_line=True):
	touch(file_path, file_name, forced_remove=forced_remove_prev)
	with open('{}/{}'.format(file_path, file_name), 'a') as f:
		f.write('{}'.format(content))
		if new_line:
			f.write('\n')

def copy_file(src_path, src_file_name, dst_path, dst_file_name):
	shutil.copyfile('{}/{}'.format(src_path, src_file_name), '{}/{}'.format(dst_path,dst_file_name))

def get_objs_in_images(pascal_anns_dir, images_name):
	image_obj = dict()
	for ix, (image_name) in enumerate(images_name):
		image_ann = ET.parse( '{}/{}.xml'.format( pascal_anns_dir, image_name) ).getroot()
		image_obj[image_name] = len( image_ann.findall('object') )
		print('image objects counter: %d/%d( %.2f %% )' % (ix+1, len(images_name), ( (ix+1)/len(images_name)*100) ), end='\r')
	print()
	return image_obj

def get_num_blocks(file_name):
	counter=0
	for ix, line in enumerate(open(file_name)):
		if line.split(' ')[-1]=='\n':
			counter+=1
	return counter

def preprocess(file_path, file_name, data_mode, saving_path, num_classes, saving_point=[0.,1.]):
	assert data_mode in ['train', 'val', 'test'], 'data_mode must be \'train\' or \'val\' or \'test\'.'

	mkdir(saving_path, data_mode)
	mkdir('{}/{}'.format( saving_path, data_mode), 'mfcc')

	num_blocks = get_num_blocks( '{}/{}'.format(file_path, file_name) )
	blocks_per_class = num_blocks // num_classes
	start_point = int(saving_point[0]*( blocks_per_class//2 ))
	end_point = int(saving_point[1]*( blocks_per_class//2 ))
	
	class_num = 0
	block_counter = 0 
	block=''
	for ix, (line) in enumerate( open( '{}/{}'.format(file_path, file_name) ) ):
		class_block = block_counter%blocks_per_class
		class_num = block_counter//blocks_per_class

		if ix==0:
			continue

		elif line.split(' ')[-1]=='\n':
			gender = 'm' if class_block<(blocks_per_class//2) else 'f'
			saving_file_name = '{}_{}_{:0=3d}.mfcc'.format(class_num, gender, class_block%(blocks_per_class//2) )
			
			valid_block = (
							start_point <= class_block and 
							class_block < end_point) or (
							blocks_per_class//2 + start_point <= class_block and 
							class_block < blocks_per_class//2 + end_point
						  )
			
			if valid_block:
				write_file( 
							'{}/{}'.format(saving_path, data_mode),
							'filenames.txt',
							'{} {}'.format(saving_file_name, class_num),
							forced_remove_prev=False,
							new_line=True
						)
				write_file(
							'{}/{}/mfcc'.format(saving_path, data_mode), 
							saving_file_name, 
							block, 
							forced_remove_prev=True,
							new_line=False)
			block_counter+=1
			block=""
		else:
			block+=line

		print('%s: %03d/%d( %.2f%% )' % ( 
										data_mode, 
										block_counter+1, 
										num_blocks, 
										( (block_counter+1)/num_blocks*100) ), 
										end='\r')
	print()

def _main():
	train_path = '.'
	train_file_name = 'Train_Arabic_Digit.txt'
	train_num_classes = 10
	train_saving_point = [0., 0.8]

	val_path = '.'
	val_file_name = 'Train_Arabic_Digit.txt'
	val_num_classes = 10
	val_saving_point = [0.8, 1.]
	
	test_path = '.'
	test_file_name = 'Test_Arabic_Digit.txt'
	test_num_classes = 10
	test_saving_point = [0, 1.]

	preprocess(train_path, train_file_name, 'train', '.', train_num_classes, saving_point=train_saving_point )
	preprocess(val_path, val_file_name, 'val', '.', val_num_classes, saving_point=val_saving_point )
	preprocess(test_path, test_file_name, 'test', '.', test_num_classes, saving_point=test_saving_point )

	

if __name__ == "__main__":
	_main()