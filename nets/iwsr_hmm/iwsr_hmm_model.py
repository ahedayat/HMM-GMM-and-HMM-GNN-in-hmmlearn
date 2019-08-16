import numpy as np
from hmmlearn import hmm

class IWSR_HMM:
    def __init__(self, 
                        transmat_prior, 
                        startprob_prior, 
                        n_iter=10, 
                        states_num=5, 
                        GMM_mix_num=3, 
                        covariance_type='diag',
                        hmm_type='gaussian' ):
        # self.hmm_model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
        #                    transmat_prior=transmat_prior, startprob_prior=startprob_prior, \
        #                    covariance_type=covariance_type, n_iter=n_iter)
        # print('**** without transmat_prior, startprob_prior')
        # self.hmm_model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
        #                    covariance_type=covariance_type, n_iter=n_iter)
        # print('**** n_mix')
        self.states_num = states_num
        self.hmm_model = None
        if hmm_type=='gaussian':
            self.hmm_model = hmm.GaussianHMM(n_components=states_num,\
                                            transmat_prior=transmat_prior, startprob_prior=startprob_prior, \
                                            covariance_type=covariance_type, n_iter=n_iter)
        elif hmm_type=='mixture':
            self.hmm_model = hmm.GMMHMM(n_components=states_num, \
                                        covariance_type=covariance_type, n_iter=n_iter)
            
        # self.hmm_model.emissionprob_ = np.ones((states_num, 1))*(1/states_num)
        # self.hmm_model.weights_ = np.ones((states_num, 1))*(1/states_num)
        # print( '{}\n{}\n{}'.format( '-'*32, self.hmm_model.weights_, '-'*32))
        # self.hmm_model.init_params = 'stw'
        # print('**** just n_componenets')
        # self.hmm_model = hmm.GMMHMM(n_components=states_num)
    
    def fit(self, data, lengths):
        self.hmm_model.fit(data, lengths)
    
    def score(self, data):
        return self.hmm_model.score(data)

    def predict(self, data):
        return self.hmm_model.predict(data)