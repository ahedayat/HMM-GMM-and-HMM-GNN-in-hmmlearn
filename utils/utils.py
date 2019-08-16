import os
import shutil
import torch
from optparse import OptionParser
# from scikits.talkbox.features import mfcc
from scipy.io import wavfile

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

def write_file(file_path, file_name, content, new_line=True, forced_remove_prev=False):
	touch(file_path, file_name, forced_remove=forced_remove_prev)
	with open('{}/{}'.format(file_path, file_name), 'a') as f:
		f.write('{}'.format(content))
		if new_line:
			f.write('\n')
		f.close()

def copy_file(src_path, src_file_name, dst_path, dst_file_name):
	shutil.copyfile('{}/{}'.format(src_path, src_file_name), '{}/{}'.format(dst_path,dst_file_name))  

def ls(dir_path):
	return os.listdir(dir_path)
    
def get_args():
    parser = OptionParser()
    parser.add_option('--analysis', dest='analysis', default=1, type='int',
                      help='analysis number')

    parser.add_option('--optimizer',type='choice', action='store', dest='optimizer',
                        choices=('adam','sgd'), default='adam', help='optimizer: ["adam", "sgd"]' )
    parser.add_option('--learning_rate', dest='lr', default=1e-3,
                      type='float', help='learning rate')
    parser.add_option('--momentum', dest='momentum', default=0.9,
                      type='float', help='momentum for sgd optimizer')
    parser.add_option('--criterion',type='choice', action='store', dest='criterion',
                        choices=('mse','cross_entropy'), default='cross_entropy', help='criterion: ["mse", "cross_entropy"]' )

    parser.add_option('--num_epochs', dest='num_epochs', default=1, type='int',
                      help='num epoch')
    parser.add_option('--start_epoch', dest='start_epoch', default=1, type='int',
                      help='start epoch')
    parser.add_option('--num_workers', dest='num_workers', default=1, type='int',
                      help='num_workers')
    parser.add_option('--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')

    parser.add_option('--preprocessing_path', dest='preprocessing_path', type='string',
                      help='preprocessing path for saving hmm results in DNN-HMM')
    parser.add_option('--preprocess', action='store_true', dest='preprocess',
                      default=False, help='preprocessing for HMM part of network')

    parser.add_option('--dataset', dest='dataset', type='string',
                      help='dataset path')
    parser.add_option('--data_size', dest='data_size', default=13, type='int',
                      help='number of cepstral coefficients')
    parser.add_option('--num_derivative', dest='num_derivative', default=2, type='int',
                      help='number of derivative of cepstral coefficients')

    parser.add_option('--n_iter', dest='n_iter', default=10, type='int',
                      help='hmm.n_iter')
    parser.add_option('--states_num', dest='states_num', default=5, type='int',
                      help='hmm.states_num')
    parser.add_option('--gmm_mix_num', dest='gmm_mix_num', default=3, type='int',
                      help='hmm.gmm_mix_num')                      
    parser.add_option('--covariance_type',type='choice', action='store', dest='covariance_type',
                        choices=('spherical','diag','full','tied'), default='diag', help='hmm.covariance_type' )                      
    parser.add_option('--hmm_type',type='choice', action='store', dest='hmm_type',
                        choices=('gaussian', 'mixture'), default='gaussian', help='hmm.hmm_type: {"gaussian", "mixture"}' )

    parser.add_option('--fnn', dest='fnn', type='string', default='100,200',
                      help='neuron num of feed-forward neural network( comma seprated )')

    (options, _) = parser.parse_args()
    return options

# def extract_mfcc(full_audio_path):
#     sample_rate, wave =  wavfile.read(full_audio_path)
#     mfcc_features = mfcc(wave, nwin=int(sample_rate * 0.03), fs=sample_rate, nceps=12)[0]
#     return mfcc_features