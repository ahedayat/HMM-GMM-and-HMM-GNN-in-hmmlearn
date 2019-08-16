import pickle
from .iwsr_hmm_model import IWSR_HMM

def iwsr_hmm_save(file_path, file_name, iwsr_hmm):
    with open('{}/{}.pkl'.format(file_path, file_name), 'wb') as file: 
        pickle.dump(iwsr_hmm, file)

def iwsr_hmm_load(file_path, file_name):
    iwsr_hmm = None
    with open('{}/{}.pkl'.format(file_path, file_name), 'rb') as file: 
        iwsr_hmm = pickle.load(file)
    return iwsr_hmm

def iwsr_hmm_train( iwsr_hmms_dict,
                    train_data):
    for ix, (mfccs, label, mfccs_length, _) in enumerate(train_data):
        label = int(label[0])
        model = iwsr_hmms_dict[label]
        print('mfcss.shape: {}, mfcss_length.shape: {}, mfccs_label: {}'.format(mfccs.shape, mfccs_length.shape, label))
        model.fit(mfccs, lengths=mfccs_length)
        print('train: %d/%d ( %.2f%% )' % (ix+1, len(train_data), (ix+1)/len(train_data)*100 ))
    print()

def iwsr_hmm_eval(  iwsr_hmms_dict,
                    eval_data):
    score_count=0
    accs = list()
    for ix, (mfccs, label, _, _) in enumerate(eval_data):
        score_list = dict()
        label = int(label[0])
        for model_label in iwsr_hmms_dict.keys():
            model = iwsr_hmms_dict[model_label]
            score = model.score(mfccs)
            score_list[model_label] = score
        predicted = max(score_list, key=score_list.get)
        if predicted==label:
            score_count+=1
        print('eval: %d/%d ( %.2f%% )' % (ix+1, len(eval_data), (ix+1)/len(eval_data)*100 ))
    print()
    print('Final recognition rate is %.2f%%' % ( score_count/len(eval_data)*100 ))