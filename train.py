import gc

import numpy as np
from sklearn.externals import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from clr import CyclicLR

import keras
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Activation, concatenate
from keras.models import Model



# GPU setup
# import tensorflow as tf
# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)


def model():
    '''
    DL model
    '''
    # first input
    inputs = Input(shape=(55, 300))
    input_head = Input(shape=(6,))

    x = Dense(256)(inputs)
    x = Activation('elu')(x)

    x = Dense(256)(x)
    x = Activation('elu')(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Dropout(0.2)(x)
    x = Flatten()(x)

    # second input
    x2 = Dense(16)(input_head)
    x2 = Activation('elu')(x2)

    x = concatenate([x, x2])
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.2)(x)

    x = Dense(64, activation='relu')(x)
    out = Dense(55, activation='linear')(x)

    model = Model(inputs=[inputs, input_head], outputs=out)

    model.compile(optimizer='nadam',
                  loss='mse',
                  )
    return model


def load_data():
    X_train_head = np.load('data/all_train_heads.npy')
    X_train = np.load('data/all_train.npy')
    Y = np.load('data/all_targets.npy')
    return X_train_head, X_train, Y

def CV_keras(X_train, X_train_head, Y):
    '''
    CV for a DL model. It saves weights from each fold.

    (np.array, np.array, np.array ) -> None
    '''

    kf = KFold(n_splits=10, random_state=23, shuffle=True)
    mse_scores = []
    for j, (train_idx, val_idx) in enumerate(kf.split(X_train_head)):
        # get a model
        mdl = model()

        debug_mode = 1

        whts_file = "./weights/weights_fold_{}.h5".format(j)

        # calbacks
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=23, verbose=debug_mode, mode='min')
        checkpoint = keras.callbacks.ModelCheckpoint(whts_file, monitor='val_loss', verbose=debug_mode,
                                                     save_best_only=True, mode='min')

        clr = CyclicLR(base_lr=0.0009, max_lr=0.03,
                       step_size=301., mode='triangular2')
        calls = [early_stopping, checkpoint, clr, keras.callbacks.TerminateOnNaN()]

        mdl.fit([X_train[train_idx], X_train_head[train_idx]], Y[train_idx], epochs=1000, batch_size=3048,
                validation_data=([X_train[val_idx], X_train_head[val_idx]], Y[val_idx]), callbacks=calls)
        # load best weights
        mdl.load_weights(whts_file)

        # save some place
        del mdl
        gc.collect()

def main():
    print('Data loading...')
    X, X_head, y = load_data()

    # scalling
    print("Scalling data...")
    X = X.reshape(-1,55*300)
    X = (X-1)*1000
    X = X.reshape(-1,55,300)

    sk = StandardScaler()
    scaler_filename = "scaler.save"
    X_head = sk.fit_transform(X_head)
    joblib.dump(sk, scaler_filename) 

    y *= 1000
    
    # train a OOF model
    print('Starting training loop...')
    CV_keras(X, X_head, y)

# start all!
main()
    
    

    
    
    
    
    
    
    
    




