''' Create generators from dataset '''

import numpy as np
import cv2
import random

HIGH_DIM = 512
GLLIM_K = 1

OUT_SIZE = 224
BATCH_SIZE = 128

def load_data_generator(rootpath, file_train, file_test, gllim=None, FT=False):
    ''' create generators from data'''

    
    def generator(rootpath, images, batch_size=BATCH_SIZE):
        
        N=len(images)
        nbatches=N/batch_size
        i=0
        while 1:
            X, Y = get_xy_from_file(rootpath, images[i*batch_size:(i+1)*batch_size])
            if(FT==False):
                yield(X, Y)
                i=i+1
                if i>=nbatches:
                    i=0
                    random.shuffle(images)
                    
            if(FT==True):
                Y_true = np.empty(len(Y), HIGH_DIM, GLLIM_K)
                for i,k,y in zip(range(len(Y)), GLLIM_K, Y):
                    Y_true[i,:,k] = np.dot(gllim.AkList[k],y) + gllim.bkList[k]
                yield(X, Y_true)
                i=i+1
                if i>=nbatches:
                    i=0
                    random.shuffle(images)


    im = open(rootpath+file_train, 'r').readlines()
    gen_train = generator(rootpath, im)
    training_size = len(im)
    
    im = open(rootpath+file_test, 'r').readlines()
    gen_test = generator(rootpath, im)
    test_size=len(im)
 
    return (gen_train,training_size), (gen_test,test_size)

def get_xy_from_file(rootpath, images, processingTarget=None):
    '''Extract data arrays from text file'''
    
    X = np.zeros((len(images),3, OUT_SIZE, OUT_SIZE), dtype=np.float32)
    Y=[]
    
    for i,image in enumerate(images):
        currentline=image.strip().split(" ")
        
        fileName=currentline[0]
        imFile = fileName
        X[i]=get_image_for_vgg(rootpath+imFile)
 
        Y.append(np.asarray(map(lambda x :float(x),currentline[1:])))

    if processingTarget:
        Y=map(processingTarget,Y)
    
    return (X, np.squeeze(np.asarray(Y)))

def get_image_for_vgg(imName):
    '''Preprocess images as VGG inputs'''
    
    im = (cv2.resize(cv2.imread(imName), (OUT_SIZE, OUT_SIZE))).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose(2,0,1)
    im = np.expand_dims(im, axis=0)
    
    return im
