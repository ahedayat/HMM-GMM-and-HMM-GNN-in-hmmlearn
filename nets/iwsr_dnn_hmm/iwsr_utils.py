import os
import pickle
import torch
import gc

import numpy as np
import utils as utility
import dataloaders.torchsad as torchsad
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable as V
from torch.utils.data import DataLoader

def iwsr_save(file_path, file_name, iwsr_hmm):
    with open('{}/{}.pkl'.format(file_path, file_name), 'wb') as file: 
        pickle.dump(iwsr_hmm, file)

def iwsr_load(file_path, file_name):
    iwsr_hmm = None
    with open('{}/{}.pkl'.format(file_path, file_name), 'rb') as file: 
        iwsr_hmm = pickle.load(file)
    return iwsr_hmm

def iwsr_accuracy(output, target, num_states, criterion):
    y = torch.argmax(output, dim=1)
    y_label = y / num_states

    y_hat = target
    if not isinstance(criterion, nn.CrossEntropyLoss):
        y_hat = torch.argmax(target, dim=1)
    y_hat_label = y_hat / num_states

    acc = y==y_hat
    acc = float( torch.sum(acc, dtype=torch.float64) / acc.size()[0] )

    label_acc = y_label==y_hat_label
    label_acc = float( torch.sum(label_acc, dtype=torch.float64) / label_acc.size()[0] )

    return label_acc, acc

def iwsr_train( iwsr,
                preprocessing_path,
                train_data,
                val_data,
                optimizer,
                criterion,
                report_path,
                train_num_derivative,
                val_num_derivative,
                num_epoch = 1,
                start_epoch = 0,
                num_workers = 1,
                gpu = False,
                preprocess=True):
    utility.mkdir(report_path, 'train_losses')
    utility.mkdir(report_path, 'train_accs')
    utility.mkdir(report_path, 'train_label_accs')

    if preprocess or not os.path.isdir('{}/train'.format(preprocessing_path)):
        iwsr_preprocess(iwsr.hmms_model, '{}/train'.format(preprocessing_path), train_data)
    states_num = iwsr.hmms_model[train_data.classes()[0]].states_num

    train_torchsad_loader = torchsad.loader( 
                                            train_data.data_root, 
                                            '{}/train'.format(preprocessing_path), 
                                            len(train_data.classes()), 
                                            states_num,
                                            train_num_derivative
                                        )


    for epoch in range(start_epoch, start_epoch+num_epoch):
        
        train_loader = DataLoader(  train_torchsad_loader,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory= gpu and torch.cuda.is_available(),
                                    num_workers=num_workers)

        print('{} epoch={} {}'.format( '#'*32, epoch, '#'*32))
        print('{} train {}'.format('-'*32, '-'*32))

        loss_item=0
        losses = list()
        accs = list()
        label_accs = list()

        for ix,(X, Y, label, states_prob) in enumerate(train_loader):
            label = int(label[0])
            X, Y, states_prob = V(X), V(Y), V(states_prob, requires_grad=False)

            X = torch.squeeze(X, dim=0)
            Y = torch.squeeze(Y, dim=0)

            if isinstance(criterion, nn.CrossEntropyLoss):
                Y = torch.argmax(Y, dim=1)

            if gpu and torch.cuda.is_available():
                X, Y, states_prob = X.cuda(), Y.cuda(), states_prob.cuda()

            output = iwsr.fnn_model(X)
            if not iwsr.fnn_model.use_softmax and not isinstance(criterion, nn.CrossEntropyLoss):
                print('softmax')
                output = F.softmax(output)
                
            states_prob = states_prob.reshape(1,-1)
            states_prob[ states_prob<1e-7 ] = float('Inf')
            output = output / (states_prob)

            prev_loss = loss_item
            loss = criterion(output,Y)
            loss_item = loss.item()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            label_acc, acc = iwsr_accuracy(output, Y, states_num, criterion)

            losses.append(loss_item)
            label_accs.append(label_acc)
            accs.append(acc)
            
            del X, Y, output, states_prob
            torch.cuda.empty_cache()
            gc.collect()

            print('ix:%d/%d( %.2f%% ), epoch=%d, label=%d, prev loss:%.3f => curr loss: %.3f, D(loss)=%.3f, label_acc=%.2f%%, acc=%.2f%%' % (   ix+1,
                                                                                                                                                len(train_data),
                                                                                                                                                (ix+1)/len(train_data)*100,
                                                                                                                                                epoch,
                                                                                                                                                label,
                                                                                                                                                prev_loss, 
                                                                                                                                                loss_item, 
                                                                                                                                                loss_item-prev_loss,
                                                                                                                                                label_acc*100,
                                                                                                                                                acc*100
                                                                                                                                )
                 )

        np.save('{}/train_losses/train_losses_epoch_{}.npy'.format(report_path, epoch), np.array(losses))
        np.save('{}/train_accs/train_accs_epoch_{}.npy'.format(report_path, epoch), np.array(accs))
        np.save('{}/train_label_accs/train_label_accs_epoch_{}.npy'.format(report_path, epoch), np.array(label_accs))

        iwsr_save('{}/models'.format(report_path), 'iwsr_dnn_hmm_epoch_{}'.format(epoch), iwsr)

        iwsr_eval(  iwsr,
                    preprocessing_path,
                    val_data,
                    criterion,
                    report_path,
                    val_num_derivative,
                    eval_mode='val',
                    num_workers=num_workers,
                    gpu=gpu,
                    epoch=epoch,
                    preprocess=preprocess
                 )
    return iwsr


def iwsr_eval(  iwsr,
                preprocessing_path,
                eval_data,
                criterion,
                report_path,
                eval_num_derivative,
                eval_mode='val',
                num_workers = 1,
                gpu = False,
                epoch=None,
                preprocess=True):

    print('{} {} {}'.format('-'*32, eval_mode, '-'*32))
    utility.mkdir(report_path, '{}_losses'.format(eval_mode))
    utility.mkdir(report_path, '{}_accs'.format(eval_mode))
    utility.mkdir(report_path, '{}_label_accs'.format(eval_mode))
    
    if preprocess or not os.path.isdir('{}/{}'.format(preprocessing_path, eval_mode)):
        iwsr_preprocess(iwsr.hmms_model, '{}/{}'.format(preprocessing_path, eval_mode), eval_data)
    
    states_num = iwsr.hmms_model[eval_data.classes()[0]].states_num

    eval_torchsad_loader = torchsad.loader( eval_data.data_root,
                                            '{}/{}'.format(preprocessing_path, eval_mode),
                                            len(eval_data.classes()), 
                                            states_num,
                                            eval_num_derivative,
                                            data_mode=eval_mode)
    eval_loader = DataLoader(   eval_torchsad_loader,
                                batch_size=1,
                                shuffle=False,
                                pin_memory= gpu and torch.cuda.is_available(),
                                num_workers=num_workers)
    losses = list()
    label_accs = list()
    accs = list()

    loss_item=0
    for ix,(X, Y, label, states_prob) in enumerate(eval_loader):
        label = int(label[0])
        X, Y, states_prob = V(X), V(Y), V(states_prob, requires_grad=False)

        X = torch.squeeze(X, dim=0)
        Y = torch.squeeze(Y, dim=0)

        if isinstance(criterion, nn.CrossEntropyLoss):
            Y = torch.argmax(Y, dim=1)

        if gpu and torch.cuda.is_available():
            X, Y, states_prob = X.cuda(), Y.cuda(), states_prob.cuda()

        output = iwsr.fnn_model(X)
        if not iwsr.fnn_model.use_softmax and not isinstance(criterion, nn.CrossEntropyLoss):
            output = F.softmax(output)

        states_prob = states_prob.reshape(1,-1)
        states_prob[ states_prob<1e-7 ] = float('Inf')
        output = output / (states_prob)

        prev_loss = loss_item
        loss = criterion(output,Y)
        loss_item = loss.item()

        label_acc, acc = iwsr_accuracy(output, Y, states_num, criterion)

        losses.append(loss_item)
        accs.append(acc)
        label_accs.append(label_acc)
        
        del X, Y, output, states_prob
        torch.cuda.empty_cache()
        gc.collect()

        print('ix=%d/%d( %.2f%% ), label=%d, prev loss:%.3f => curr loss:%.3f, D(loss)=%.3f, acc=%.2f' % (  ix+1,
                                                                                                            len(eval_data),
                                                                                                            (ix+1)/len(eval_data)*100,
                                                                                                            label,
                                                                                                            prev_loss, 
                                                                                                            loss_item, 
                                                                                                            loss_item-prev_loss,
                                                                                                            acc*100
                                                                                               )

        # print('ix=%d, prev loss:%.3f => curr loss:%.3f, D(loss)=%.3f, label_acc=%.2f, acc=%.2f' % ( ix,
        #                                                                                             prev_loss, 
        #                                                                                             loss_item, 
        #                                                                                             loss_item-prev_loss,
        #                                                                                             label_acc*100,
        #                                                                                             acc*100
        #                                                                                           )
             )

    np.save('{}/{}_losses/{}_losses{}.npy'.format(report_path, eval_mode, eval_mode, '_epoch_{}'.format(epoch) if epoch is not None else ''), np.array(losses))
    np.save('{}/{}_accs/{}_accs{}.npy'.format(report_path, eval_mode, eval_mode, '_epoch_{}'.format(epoch) if epoch is not None else ''), np.array(accs))
    np.save('{}/{}_label_accs/{}_label_accs{}.npy'.format(report_path, eval_mode, eval_mode, '_epoch_{}'.format(epoch) if epoch is not None else ''), np.array(label_accs))

def iwsr_preprocess(iwsr_hmms_dict,
                    preprocessing_path,
                    data_loader):

    utility.mkdir(preprocessing_path, 'mfccs_preprocessed', forced_remove=True)
    utility.mkdir(preprocessing_path, 'states_prob', forced_remove=True)

    all_frames = dict()
    state_frames = dict()
    num_classes = len(data_loader.classes())
    first_class = data_loader.classes()[0]
    states_num = iwsr_hmms_dict[first_class].states_num

    for ix, (mfccs, label, mfcss_length, mfccs_filename) in enumerate(data_loader):
        label = int(label[0])
        
        all_states_prob = np.zeros((num_classes, states_num), dtype=np.float64)
        one_hot_vec = np.zeros( (mfccs.shape[0], states_num*num_classes), dtype=np.float64)
        
        for jx,(_class) in enumerate(iwsr_hmms_dict.keys()):
            model = iwsr_hmms_dict[_class]
            predicted_states = model.predict( mfccs )
            for kx in range(predicted_states.shape[0]):
                state = predicted_states[kx]
                all_states_prob[ jx, state ] += 1
            all_states_prob[jx, :] /= predicted_states.shape[0]

            if _class==label:
                for kx in range(predicted_states.shape[0]):
                    state_index = label*states_num + predicted_states[kx]
                    one_hot_vec[kx, state_index] = 1

        np.save('{}/mfccs_preprocessed/{}.npy'.format(preprocessing_path, mfccs_filename), one_hot_vec)
        np.save('{}/states_prob/{}.npy'.format(preprocessing_path, mfccs_filename), all_states_prob)

        print('mfccs(%s): %d/%d( %.2f%% )' % (preprocessing_path, ix+1, len(data_loader), (ix+1)/len(data_loader)*100 ) )

        # model = iwsr_hmms_dict[label]
        # probable_states = model.predict( mfccs )

        # states_num = 3
        # one_hot_vec = np.zeros( ( probable_states.shape[0], states_num*num_classes ), dtype=np.float)
        # for num in range(one_hot_vec.shape[0]):
        #     state_index = label*states_num+ probable_states[num]
        #     one_hot_vec[num, state_index] = 1.

        # if label not in all_frames:
        #     all_frames[label] = 0
        # all_frames[label] += mfccs.shape[0]

        # if label not in state_frames:
        #     state_frames[label] = [0]*states_num
        # for num in range(probable_states.shape[0]):
        #     state = int(probable_states[num])
        #     state_frames[label][state] = state_frames[label][state]+1

        # np.save('{}/mfccs_preprocessed/{}.npy'.format(preprocessing_path, mfccs_filename), one_hot_vec)

        # utility.write_file(preprocessing_path, 'filenames.txt', mfccs_filename)

    # for label in all_frames.keys():
    #     states_prob = np.zeros( ( len(state_frames[label]), ) )
    #     for ix, (state_frame_num) in enumerate(state_frames[label]):
    #         states_prob[ix] = state_frame_num/all_frames[label]
    #     print('saving.... {}/states_prob/{}.npy'.format(preprocessing_path, label))
    #     np.save( '{}/states_prob/{}.npy'.format(preprocessing_path, label), states_prob )
