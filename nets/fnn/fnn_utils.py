import torch

def fnn_save( file_path, file_name, model, optimizer=None ):
    state_dict = {
        'net_arch' : 'feed_forward',
        'model' : model.state_dict(),
    }
    if optimizer is not None:
        state_dict[ 'optimizer' ] = optimizer.state_dict()

    torch.save( state_dict, '{}/{}.pth'.format(file_path,file_name) )

def fnn_load(file_path, file_name, model, optimizer=None):
    check_points = torch.load('{}/{}.pth'.format(file_path,file_name))
    keys = check_points.keys()


    assert ('net_arch' in keys) and ('model' in keys), 'Cannot read this file in address : {}/{}.pth'.format(file_path,file_name)
    assert check_points['net_arch']=='feed_forward', 'This file model architecture is not \'resnet\''
    model.load_state_dict( check_points['model'] )
    if optimizer is not None:
        optimizer.load_state_dict(check_points['optimizer'])
    return model, optimizer

def fnn_batch_train(model, optimizer, criterion, X, Y ):
    assert len(X.size())==4, 'Error: expected 4D input.'
    assert X.size()[0]==Y.size()[0], 'Error: input and target must have same size in first dimension.'
    batch_size = X.size()[0]

    output = model(X)
    loss = criterion(output, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return model, loss.item()

def fnn_batch_eval(model, criterion, X, Y ):
    assert len(X.size())==4, 'Error: expected 4D input.'
    assert X.size()[0]==Y.size()[0], 'Error: input and target must have same size in first dimension.'
    batch_size = X.size()[0]
    
    output = model(X)
    loss = criterion(output, Y)

    return model, loss.item()