import numpy as np
from sklearn.model_selection import train_test_split
from model import RCGAN
import os
import argparse

FILE_NAME = 'inputs/sin_wave.npz'
#FILE_NAME = 'inputs/mnist1.npz'
SEED = 12345

def seq_mnist_normalize(data):
    """
    Normalize for rot_mnist
    """
    def MaxMinNorm(data):
        return ( ( (data - data.min()) / (data.max() - data.min()) ) * 2 - 1 ).tolist()

    ret_rot_data = []
    for data in data:
        ret_rot_data.append( 
            list(
                map(
                    lambda seq_data: MaxMinNorm(seq_data), 
                    data
                )
            )
        )

    return np.array(ret_rot_data)

def seq_bp_normalize(data):
    """
    Normalize for rot_mnist
    """
    from sklearn.preprocessing import MinMaxScaler
    mms = MinMaxScaler()
    tmp = data.reshape( data.shape[0], data.shape[1] * data.shape[2] )
    tmp = mms.fit_transform(tmp) * 2 - 1
    print("Save scale")
    np.savez('bp_data_mms.npz',data_min = mms.data_min_, data_max = mms.data_max_)
    
    tmp = tmp.reshape( data.shape[0], data.shape[1], data.shape[2] )
    
    return tmp

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-inputs', default=FILE_NAME)
    args = parser.parse_args()
    
    # load data
    ndarr = np.load(args.inputs)
    X_train, X_eval, y_train, y_eval = train_test_split(ndarr['x'],
                                                        ndarr['y'],
                                                        test_size=0.1,
                                                        random_state=SEED)

    assert X_train.ndim == 3, 'x shape is expected 3 dims, but {} shapes'.format(
        X_train.ndim)

    if args.inputs == 'inputs/mnist1.npz':
        X_train = seq_mnist_normalize(X_train) 
        X_eval = seq_mnist_normalize(X_eval)
    elif args.inputs == 'inputs/bp_data.npz':
        X_eval = seq_bp_normalize(X_eval)
        X_train = seq_bp_normalize(X_train) 
    
    print('train x shape:', X_train.shape)

    # hyper parameter for training
    args = {}
    args['seq_length'] = X_train.shape[1]
    args['input_dim'] = X_train.shape[2]
    args['latent_dim'] = 500
    args['hidden_dim'] = 500
    args['embed_dim'] = 10
    args['n_epochs'] = 100
    args['batch_size'] = 64
    args['num_classes'] = len(np.unique(y_train))
    args['save_model'] = True
    args['instance_noise'] = False
    args['oneside_smooth'] = True
    args['label_flipping'] = 0.05
    args['dp_sgd'] = True
    args['sigma'] = 0.1
    args['l2norm_bound'] = 0.1
    args['learning_rate'] = 0.1
    args['total_examples'] = X_train.shape[0]
    
    if not os.path.isdir('models') and args['save_model']:
        os.mkdir('models')
        print('make directory for save models')

    rcgan = RCGAN(**args)

    rcgan.train(args['n_epochs'],
                X_train,
                y_train,
                X_eval,
                y_eval)


if __name__ == '__main__':
    # choose GPU devise
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
