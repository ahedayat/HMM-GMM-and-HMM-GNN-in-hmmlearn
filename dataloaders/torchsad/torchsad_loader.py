import os
import torch
import numpy as np
from random import shuffle
import utils as utilities
from .torchsad_utils import read_mfcc_file
from torch.utils.data import Dataset

class TorchSADLoader(Dataset):
    """
        data_root/
                 |_mfccs_raw/
                 |_mfccs_preprocessed/
                 |_states_prob
                 |_filenames.txt
    """
    def __init__(self, raw_data_root, preprocessed_data_root, num_classes, num_states, num_derivative, data_mode='train', shuffle_data=True):
        self.raw_data_root = raw_data_root
        self.preprocessed_data_root = preprocessed_data_root
        self.raw_mfccs_path = '{}/mfcc'.format(self.raw_data_root)
        self.preprocessed_mfccs_path = '{}/mfccs_preprocessed'.format(self.preprocessed_data_root)
        self.states_prob_path = '{}/states_prob'.format(self.preprocessed_data_root)
        self.num_classes = num_classes
        self.num_states = num_states
        self.data_mode = data_mode
        self.num_derivative = num_derivative
        
        self.file_infos = [ (line.split()[0], int(line.split()[1])) for line in open('{}/filenames.txt'.format(self.raw_data_root) ) ]
        if shuffle_data:
            shuffle(self.file_infos)

        # state_prob_filenames = utilities.ls( '{}/states_prob'.format(self.preprocessed_data_root) )
        # self.states_prob = dict()
        # for state_prob_filename in state_prob_filenames:
        #     label = int( os.path.splitext( state_prob_filename.split('/')[-1] )[0] )
        #     self.states_prob[label] = np.load( '{}/states_prob/{}'.format(self.preprocessed_data_root, state_prob_filename) )

        # self.prob_states = torch.zeros(self.num_classes*self.num_states, dtype=torch.double)
        # for label in range(num_classes):
        #     self.prob_states[ label*self.num_states:(label+1)*self.num_states ] = torch.tensor( self.states_prob[label], dtype=torch.double )

    def __getitem__(self, ix):
        file_name, file_class = self.file_infos[ix]
        
        raw_mfccs = read_mfcc_file('{}'.format(self.raw_mfccs_path), '{}'.format(file_name))
        mfccs = np.zeros( (raw_mfccs.shape[0], (self.num_derivative+1)*raw_mfccs.shape[1]) )
        mfccs[:, :raw_mfccs.shape[1]] = raw_mfccs
        for ix in range( mfccs.shape[0] ):
            for i in range(1,self.num_derivative+1):
                mfccs[ix, (i-1)*raw_mfccs.shape[1]:i*raw_mfccs.shape[1]] = np.gradient(raw_mfccs[ix,:], i)

        label = np.array([file_class])

        target=torch.tensor([])
        target = np.load( '{}/{}.mfcc.npy'.format(self.preprocessed_mfccs_path, os.path.splitext( file_name.split('/')[-1] )[0]) )
        target = torch.tensor( target, dtype=torch.double )

        states_prob = np.load('{}/{}.mfcc.npy'.format(self.states_prob_path, os.path.splitext( file_name.split('/')[-1] )[0]) )
        states_prob = torch.tensor( states_prob, dtype=torch.double)
        
        mfccs = torch.tensor( mfccs, dtype=torch.double )
        label = torch.tensor( label, dtype=torch.double )

        return mfccs, target, label, states_prob

    def __len__(self):
        return len( self.file_infos )