import numpy as np
from random import shuffle
from .sad_utils import read_mfcc_file

class SADLoader:
    def __init__(self, data_root, num_derivative=2):
        self.data_root = data_root
        self.num_derivative = num_derivative
        self.mfccs_path = '{}/mfcc'.format( self.data_root )

        self.file_infos = [ (line.split()[0], int(line.split()[1])) for line in open('{}/filenames.txt'.format(self.data_root) ) ]
        
        # self.class_dict = dict()
        self.data_classes = list()
        for ix, (_, file_class) in enumerate(self.file_infos):
            if file_class not in self.data_classes:
                self.data_classes.append(file_class)
        #     if file_class not in self.class_dict.keys():
        #         self.class_dict[file_class] = []
        #     self.class_dict[file_class].append(file_name)
            


    # def __getitem__(self, ix):
    #     assert ix in self.data_classes, '{} class does not exist.'.format(ix)

    #     mfccs = np.zeros( (len(self.class_dict[ix]), self.num_derivative*self.data_size) )
    #     length = np.zeros( (len(self.class_dict[ix]),) )

    #     for ix, (file_name) in enumerate(self.class_dict[ix]):
    #         x = read_mfcc_file( self.mfccs_path, file_name)
    #         mfccs[ix, 0:self.data_size] = x
    #         for i in range(self.num_derivative):
    #             mfccs[ix, (i-1)*self.data_size:i*self.data_size] = np.gradient(x, i)

    #     # mfccs = np.vstack(mfccs)

    #     return mfccs

    def __getitem__(self, ix, get_file_name=False):
        file_name, file_class = self.file_infos[ix]
        raw_mfccs = read_mfcc_file('{}'.format(self.mfccs_path), '{}'.format(file_name))
        label = np.array([file_class])

        mfccs = np.zeros( (raw_mfccs.shape[0], (self.num_derivative+1)*raw_mfccs.shape[1]) )
        mfccs[:, :raw_mfccs.shape[1]] = raw_mfccs
        for ix in range( mfccs.shape[0] ):
            for i in range(1,self.num_derivative+1):
                mfccs[ix, (i-1)*raw_mfccs.shape[1]:i*raw_mfccs.shape[1]] = np.gradient(raw_mfccs[ix,:], i)

        mfccs_length = np.ones((mfccs.shape[0],), dtype=np.int)*1
        # mfccs_length = mfccs_length.T
        # mfccs = np.hstack(mfccs)
        if get_file_name:
            return mfccs, label, mfccs_length, file_name

        return mfccs, label, mfccs_length, file_name

    def shuffle(self):
        shuffle(self.file_infos)
        # for class_name in self.class_dict.keys():
        #     shuffle( self.class_dict[class_name] )
    
    def classes(self):
        return self.data_classes

    def __len__(self):
        return len( self.file_infos )