import os
import warnings
import torch
import torch.nn as nn
import numpy as np
import utils as utility
import dataloaders.sad as sad
import nets.iwsr_dnn_hmm as iwsr
import torch.optim as optim
import nets.iwsr_hmm as hmm
import nets.fnn as fnn

def _main(args):

    warnings.filterwarnings("ignore") 

    ##### Creating Analysis Saving Path #####
    analysis_num = args.analysis
    if os.path.isdir('./reports'):
        utility.mkdir('.', 'reports', forced_remove=False)
    utility.mkdir('./reports', '{}'.format(analysis_num), forced_remove=False)
    utility.mkdir('./reports/{}'.format(analysis_num), 'models', forced_remove=False)
    reports_path = './reports/{}'.format(analysis_num)

    ##### Constructing Data Loader #####
    train_data_root, train_num_derivative = ('{}/train'.format(args.dataset), 2)
    train_data_loader = sad.loader( data_root=train_data_root,\
                                    num_derivative=train_num_derivative)
    train_classes = train_data_loader.classes()

    val_data_root, val_num_derivative = ('{}/val'.format(args.dataset), 2)
    val_data_loader = sad.loader(   data_root=val_data_root,
                                    num_derivative=val_num_derivative
                                )

    ##### Constructing HMMs #####
    models_path = './reports/3/models'
    hmms = dict()
    for _class in train_classes:
        hmms[_class] = hmm.load(models_path, 'iwsr_hmm_{}'.format(_class))
        # hmms[_class] = None

    ##### Constructing NNs #####
    input_size, output_size = (13*(train_num_derivative+1), len(train_classes)*3)
    layers_neuron_num = [ int(num_neuron) for num_neuron in args.fnn.split(',') ] 
    layers_neuron_num = [input_size] + layers_neuron_num + [output_size]
    net = fnn.model( layers_neuron_num,
                    soft_max=True if args.criterion=='mse' else False).double()
    if args.gpu and torch.cuda.is_available():
        net = net.cuda()

    ##### Constructing DNN-HMM #####
    model = iwsr.model(hmms, net)
    if args.start_epoch!=0:
        model = iwsr.load('{}/models'.format(reports_path), 'iwsr_dnn_hmm_epoch_{}'.format(args.start_epoch-1))

    ##### Constructing Training Requirements #####
    optimizer = None
    if args.optimizer=='sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer=='adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

    criterion = None
    if args.criterion=='mse':
        criterion = nn.MSELoss()
    elif args.criterion=='cross_entropy':
        criterion = nn.CrossEntropyLoss()
    
    ##### Training #####
    if not os.path.isdir(args.preprocessing_path):
        utility.mkdir(args.preprocessing_path, 'analysis_{}'.format(analysis_num), forced_remove=False)
    model = iwsr.train( model,
                        preprocessing_path='{}/analysis_{}'.format(args.preprocessing_path, analysis_num),
                        preprocess=args.preprocess,
                        train_data=train_data_loader,
                        val_data=val_data_loader,
                        optimizer=optimizer,
                        criterion=criterion,
                        report_path=reports_path,
                        num_epoch=args.num_epochs,
                        start_epoch=args.start_epoch,
                        num_workers=args.num_workers,
                        gpu=args.gpu and torch.cuda.is_available(),
                        train_num_derivative=train_num_derivative,
                        val_num_derivative=val_num_derivative)

    #### Saving Models ####
    iwsr.save('./reports/{}/models'.format(analysis_num), 'iwsr_dnn_hmm', model)
    # for train_class in train_classes:
    #     iwsr.save('./reports/{}/models'.format(analysis_num), 'iwsr_dnn_hmm_{}'.format(train_class), iwsr)



if __name__ == "__main__":
    args = utility.get_args()
    _main(args)