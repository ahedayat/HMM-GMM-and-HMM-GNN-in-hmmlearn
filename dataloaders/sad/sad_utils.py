import numpy as np

def read_mfcc_file(file_path, file_name):
    mfcc_coefficient = list()
    for line in open('{}/{}'.format( file_path, file_name ) ):
        mfcc_coefficient.append( [ float(coefficient) for coefficient in line.split()] )

    return np.array( mfcc_coefficient )