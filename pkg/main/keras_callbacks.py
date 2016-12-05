import keras.callbacks
from keras import backend as k

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class ValLossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_loss'))

class decay_lr(keras.callbacks.Callback):
    ''' 
        n_epoch = no. of epochs after decay should happen.
        decay = decay value
    '''  
    def __init__(self, n_epoch, decay):
        super(decay_lr, self).__init__()
        self.n_epoch=n_epoch
        self.decay=decay

    def on_epoch_begin(self, epoch, logs={}):
        old_lr = self.model.optimizer.lr.get_value()
        if epoch > 1 and epoch%self.n_epoch == 0 : # decay start after 50 epoch
            new_lr= self.decay*old_lr
            k.set_value(self.model.optimizer.lr, new_lr)
            print "New learning rate : ", new_lr
        else:
            k.set_value(self.model.optimizer.lr, old_lr)
