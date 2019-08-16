import os
import warnings
import numpy as np
import utils as utility
import dataloaders.sad as sad
import nets.iwsr_hmm as iwsr

def _main(args):

    warnings.filterwarnings("ignore") 

    ##### Creating Analysis Saving Path #####
    analysis_num = args.analysis
    if os.path.isdir('./reports'):
        utility.mkdir('.', 'reports', forced_remove=False)
    utility.mkdir('./reports', '{}'.format(analysis_num), forced_remove=True)
    utility.mkdir('./reports/{}'.format(analysis_num), 'models', forced_remove=True)

    ##### Constructing Data Loader #####
    train_data_root, train_data_size, train_num_derivative = (args.dataset, args.data_size, args.num_derivative)
    train_data_loader = sad.loader( data_root=train_data_root,\
                                    num_derivative=train_num_derivative)
    train_classes = train_data_loader.classes()

    ##### Constructing Model #####
    hmm_n_iter, hmm_states_num, hmm_GMM_mix_num, hmm_covariance_type, hmm_type= (   args.n_iter, 
                                                                                    args.states_num, 
                                                                                    args.gmm_mix_num, 
                                                                                    args.covariance_type,
                                                                                    args.hmm_type)
    # hmm_tmp_p = 1.0/(hmm_states_num-2+1e-7)
    # hmm_mean = []
    # hmm_covariance = 
    
    # hmm_transmat_prior = np.array([ [hmm_tmp_p, hmm_tmp_p, hmm_tmp_p, 0 ,0], \
    #                                 [0, hmm_tmp_p, hmm_tmp_p, hmm_tmp_p , 0], \
    #                                 [0, 0, hmm_tmp_p, hmm_tmp_p,hmm_tmp_p], \
    #                                 [0, 0, 0, 0.5, 0.5], \
    #                                 [0, 0, 0, 0, 1]],dtype=np.float)
    # hmm_transmat_prior = np.diag( np.diag( np.random.randn( hmm_states_num, hmm_states_num) ) )
    # hmm_startprob_prior = np.array([0.5, 0.5, 0, 0, 0],dtype=np.float)
    # hmm_startprob_prior = np.random.randn(hmm_states_num)
    hmm_startprob_prior = np.zeros((hmm_states_num,), dtype=np.float)
    hmm_startprob_prior[0] = 1.
    hmm_transmat_prior = np.ones((hmm_states_num, hmm_states_num), dtype=np.float)*(1/hmm_states_num)

    hmms = dict()
    for train_class in train_classes:
        hmms[train_class] = iwsr.model( transmat_prior=hmm_transmat_prior,
                                        startprob_prior=hmm_startprob_prior,
                                        n_iter=hmm_n_iter,
                                        states_num=hmm_states_num,
                                        GMM_mix_num=hmm_GMM_mix_num,
                                        covariance_type=hmm_covariance_type,
                                        hmm_type=hmm_type)

    ##### Training #####
    iwsr.train(hmms, train_data_loader)

    #### Saving Models ####
    for train_class in train_classes:
        iwsr.save('./reports/{}/models'.format(analysis_num), 'iwsr_hmm_{}'.format(train_class), hmms[train_class])



if __name__ == "__main__":
    args = utility.get_args()
    _main(args)