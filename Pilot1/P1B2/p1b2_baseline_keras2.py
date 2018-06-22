from __future__ import print_function

import numpy as np
import tensorflow as tf
import random as rn

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import Callback, ModelCheckpoint
from keras.regularizers import l2

import p1b2


BATCH_SIZE = 64
NB_EPOCH = 1#20                 # number of training epochs
PENALTY = 0.00001             # L2 regularization penalty
ACTIVATION = 'sigmoid'
FEATURE_SUBSAMPLE = None
DROP = None

L1 = 1024
L2 = 512
L3 = 256
L4 = 0
LAYERS = [L1, L2, L3, L4]


class BestLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.best_val_loss = np.Inf
        self.best_val_acc = -np.Inf
        self.best_model = None

    def on_epoch_end(self, batch, logs={}):
        if float(logs.get('val_loss', 0)) < self.best_val_loss:
            self.best_model = self.model
        self.best_val_loss = min(float(logs.get('val_loss', 0)), self.best_val_loss)
        self.best_val_acc = max(float(logs.get('val_acc', 0)), self.best_val_acc)


def extension_from_parameters():
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.A={}'.format(ACTIVATION)
    ext += '.B={}'.format(BATCH_SIZE)
    ext += '.D={}'.format(DROP)
    ext += '.E={}'.format(NB_EPOCH)
    if FEATURE_SUBSAMPLE:
        ext += '.F={}'.format(FEATURE_SUBSAMPLE)
    for i, n in enumerate(LAYERS):
        if n:
            ext += '.L{}={}'.format(i+1, n)
    ext += '.P={}'.format(PENALTY)
    return ext


def main(X_train = None, y_train = None, X_test = None, y_test = None, DeterministicResults = False):
    if(DeterministicResults):
        __setSession()

    if X_train is None:
        (X_train, y_train), (X_test, y_test) = p1b2.load_data(n_cols=FEATURE_SUBSAMPLE)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = Sequential()

    model.add(Dense(LAYERS[0], input_dim=input_dim,
                    activation=ACTIVATION,
                    kernel_regularizer=l2(PENALTY),
                    activity_regularizer=l2(PENALTY)))

    for layer in LAYERS[1:]:
        if layer:
            if DROP:
                model.add(Dropout(DROP))
            model.add(Dense(layer, activation=ACTIVATION,
                            kernel_regularizer=l2(PENALTY),
                            activity_regularizer=l2(PENALTY)))

    model.add(Dense(output_dim, activation=ACTIVATION))
    

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())

    ext = extension_from_parameters()
    checkpointer = ModelCheckpoint(filepath='model'+ext+'.h5', save_best_only=True)
    history = BestLossHistory()

    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=NB_EPOCH,
              validation_split=0.2,
              callbacks=[history, checkpointer])

    y_pred = history.best_model.predict(X_test)

    print('best_val_loss={:.5f} best_val_acc={:.5f}'.format(history.best_val_loss, history.best_val_acc))
    print('Best model saved to: {}'.format('model'+ext+'.h5'))

    scores = p1b2.evaluate(y_pred, y_test)
    print('Evaluation on test data:', scores)

    submission = {'scores': scores,
                  'model': model.summary(),
                  'submitter': 'Developer Name' }

    # print('Submitting to leaderboard...')
    # leaderboard.submit(submission)
    __resetSeed()
    return history.best_model

def __resetSeed():
    np.random.seed()
    rn.seed()

def __setSession():
    # Sets session for deterministic results
    # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

    
    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(42)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)
    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    from keras import backend as K
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    #tf.global_variables_initializer()
    tf.set_random_seed(1234)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


if __name__ == '__main__':
    main()
