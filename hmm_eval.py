import os
import warnings
import numpy as np
import utils as utility
import dataloaders.sad as sad
import nets.iwsr_hmm as iwsr

def _main(args):

    warnings.filterwarnings("ignore") 

    ##### Analysis Number #####
    analysis_num = args.analysis

    ##### Constructing Data Loader #####
    eval_data_root, eval_num_derivative = (args.dataset,  args.num_derivative)
    eval_data_loader = sad.loader( data_root=eval_data_root,\
                                    num_derivative=eval_num_derivative)
    eval_classes = eval_data_loader.classes()

    ##### Load Models #####
    loading_path = './reports/{}/models'.format(analysis_num)
    hmms = dict()
    for eval_class in eval_classes:
        hmms[eval_class] = iwsr.load(loading_path, 'iwsr_hmm_{}'.format(eval_class))

    ##### Evaluating #####
    iwsr.eval(hmms, eval_data_loader)

if __name__ == "__main__":
    args = utility.get_args()
    _main(args)