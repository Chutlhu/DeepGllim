'''Import modules'''
import numpy as np
import time
import sys
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras_callbacks import LossHistory, ValLossHistory
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from VGG16_sequential import VGG16, extract_XY_generator
from gllim  import GLLIM
from data_generator import load_data_generator
from prob_generator import resp_generators
from test import run_eval
from plot_keypoints import show_keypoints
from PCA_init import add_pca_layer
np.set_printoptions(threshold=np.nan)

ROOTPATH ='/path/to/data'

FEATURES_SIZE = 512
HIGH_DIM = FEATURES_SIZE
LOW_DIM = 10 #to modify according to the task
GLLIM_K = 1
MAX_ITER_EM = 100
ITER = 2
WIDTH = 224
PB_FLAG = 'keypoints' #to modify according to the task

BATCH_SIZE = 128
NB_EPOCH = 3
LEARNING_RATE = 1e-01

class DeepGllim:
    ''' Class of deep gllim model'''

    def __init__(self, k, PCA=None):

        self.k = k
        self.PCA = PCA
        self.gllim = GLLIM(self.k, HIGH_DIM, LOW_DIM)
        self.network = VGG16(weights='imagenet')


    def fit(self, (generator_training, n_train),
            learning_rate=0.1, it=2, f=sys.argv[1]):
        '''Trains the model for a fixed number of epochs and iterations.
           # Arguments
                X_train: input data, as a Numpy array or list of Numpy arrays
                    (if the model has multiple inputs).
                Y_train : labels, as a Numpy array.
                batch_size: integer. Number of samples per gradient update.
                learning_rate: float, learning rate
                nb_epoch: integer, the number of epochs to train the model.
                validation_split: float (0. < x < 1).
                    Fraction of the data to use as held-out validation data.
                validation_data: tuple (x_val, y_val) or tuple
                    (x_val, y_val, val_sample_weights) to be used as held-out
                    validation data. Will override validation_split.
                it: integer, number of iterations of the algorithm
                f: text file for responsability trick
                

            # Returns
                A `History` object. Its `History.history` attribute is
                a record of training loss values and metrics values
                at successive epochs, as well as validation loss values
                and validation metrics values (if applicable).
            '''
        start_time_training = time.time()

        print "Training Deep Gllim"

        features_training, target_training = extract_XY_generator(self.network, generator_training, n_train)
        print "features size:", features_training.shape
        add_pca_layer(self.network, features_training, self.PCA)
        self.network.add(BatchNormalization())
        self.network.summary()
        
        for i in range(it):
            
            # Extract features            
            features_training, target_training = extract_XY_generator(self.network, generator_training, n_train)
            #fit gllim
            self.gllim.fit(target_training, features_training, MAX_ITER_EM, (i == 0), None)
            
            # Fine-tunning
            prec = 1/self.gllim.SigmakSquareList[0] 
            DeepGllim.fine_tune(self, 26, learning_rate, f)
    
        features_training, target_training = extract_XY_generator(self.network, generator_training, n_train)
        
        self.gllim.fit(target_training, features_training, MAX_ITER_EM, False, None)
        self.gllim.inversion()       
        
        print "--- %s seconds for training Deep Gllim---" % (time.time() - start_time_training)

    def fine_tune(self, layer_nb, learning_rate, data_file):
        '''Fine tune the network according to our custom loss function'''
        
        (generator, N_TRAIN), (generator_val, N_VAL) = resp_generators(ROOTPATH, data_file,
                                                                       self.gllim,
                                                                       batch_size=BATCH_SIZE)

        # train only some layers
        for layer in self.network.layers[:layer_nb]:
            layer.trainable = False
        for layer in self.network.layers[layer_nb:]:
            layer.trainable = True
        self.network.layers[-1].trainable = True

        # compile the model
        sgd = SGD(lr=learning_rate,
                  momentum=0.9,
                  decay=1e-06,
                  nesterov=True)

        self.network.compile(optimizer=sgd,
                             loss='mse',
                             metrics=['accuracy'])

        self.network.summary()


        checkpointer = ModelCheckpoint(filepath="/services/scratch/perception/rjuge/Deep_Gllim_"+PB_FLAG+"_K"+str(GLLIM_K)+"_weights.hdf5",
                                       monitor='val_loss',
                                       verbose=1,
                                       save_weights_only=True,
                                       save_best_only=True,
                                       mode='min')


        loss_history = LossHistory()
        val_loss_history = ValLossHistory()

        # train the model on the new data for a few epochs
        self.network.fit_generator(generator,
                                   samples_per_epoch=N_TRAIN,
                                   nb_epoch=NB_EPOCH,
                                   verbose=1,
                                   callbacks=[checkpointer,
                                              loss_history,
                                              val_loss_history],
                                   validation_data=generator_val,
                                   nb_val_samples=N_VAL)

        self.network.load_weights("/services/scratch/perception/rjuge/Deep_Gllim_"+PB_FLAG+"_K"+str(GLLIM_K)+"_weights.hdf5")

        return self.network

    def predict(self, (generator, n_predict)):
        '''Generates output predictions for the input samples,
           processing the samples in a batched way.
        # Arguments
            generator: input a generator object.
            batch_size: integer.
        # Returns
            A Numpy array of predictions.
        '''
        
        features_test, _ = extract_XY_generator(self.network, generator, n_predict)
        gllim_predict = self.gllim.predict_high_low(features_test)

        return gllim_predict
    
    def evaluate(self, (generator, n_eval), l=WIDTH, pbFlag=PB_FLAG):
        '''Computes the loss on some input data, batch by batch.

        # Arguments
            generator: input a generator object.
            batch_size: integer. Number of samples per gradient update.

        # Returns
            Scalar test loss (if the model has no metrics)
            or list of scalars (if the model computes other metrics).
            The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        '''
        
        features_test, target_test = extract_XY_generator(self.network, generator, n_eval)
        gllim_predict = self.gllim.predict_high_low(features_test)
        run_eval(gllim_predict, target_test, l, pbFlag)

if __name__ == '__main__':

    deep_gllim = DeepGllim(k=GLLIM_K, PCA=FEATURES_SIZE)

    train_txt = sys.argv[1]
    test_txt = sys.argv[2]

    (gen_training, N_train), (gen_test, N_test) = load_data_generator(ROOTPATH, train_txt, test_txt, gllim=None, FT=False)
    
    deep_gllim.fit((gen_training, N_train),
                   learning_rate=LEARNING_RATE,
                   it=ITER, f=train_txt)

    predictions = deep_gllim.predict((gen_test, N_test))
    
    deep_gllim.evaluate((gen_test, N_test), WIDTH, PB_FLAG)
