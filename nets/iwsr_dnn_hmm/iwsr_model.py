class IWSR_DNN_HMM:
    def __init__(self, hmms_model, fnn_model):
        self.hmms_model = hmms_model
        self.fnn_model = fnn_model
    
    def predict(self, x, label):
        x = self.hmms_model[label].predict(x)
        return x

    def cuda(self):
        self.fnn_model = self.fnn_model.cuda()
        return self