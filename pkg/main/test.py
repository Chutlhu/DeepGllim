import numpy as np
from sklearn.metrics import mean_absolute_error, explained_variance_score

def run_eval(Y_pred, Y_true, l, pbFlag):
    print "Evaluating"
    if (pbFlag != 'keypoints'):
       
        # mean squared error
        err = np.sqrt(np.sum((Y_pred-Y_true)**2, axis=1))
        print err
        
        MSE = np.mean(err)
        print('Mean Squared Error:', MSE)
        
        # mean absolute error
        MAE = mean_absolute_error(Y_true, Y_pred, sample_weight=None, multioutput='raw_values')
        evs = explained_variance_score(Y_true, Y_pred, sample_weight=None, multioutput='raw_values')
        
        # Head pose estimation: pitch, yaw, roll
        print('Mean absolute error:', MAE)
        print('Explained variances score:', evs)
        
    elif (pbFlag == 'keypoints'):
        # We need to change the shape of the Y_pred and Y_true matrices because the evaluation is different than in Biwi
        print Y_pred.shape
        Y_pred2 = np.reshape(Y_pred, (5*Y_pred.shape[0],2), order='C')
        Y_true2 = np.reshape(Y_true, (5*Y_true.shape[0],2), order='C')
        
        # mean squared error
        err = np.sqrt(np.sum((Y_pred2-Y_true2)**2, axis=1))
        
        # Facial landmark detection performance
        # Performance is measured with the average  detection  error  and  the  failure  rate  of  each  facial
        # point. They indicate the accuracy and reliability of an algorithm. The detection error is measured as
        # MSE divided by the width of the bounding box. If an error is larger than 5%, it is counted as failure.
        # From http://www.ee.cuhk.edu.hk/~xgwang/papers/sunWTcvpr13.pdf
        # Facial landmark detection: LE, RE, N, LM, RM
        listErr = np.empty((5,1))
        listFailures = np.empty((5,1))
        for i in range(5):
            temp = 0
            tempFailures = 0
            for j in xrange(i,len(err),5):
                temp += (err[j]/float(l))
                # If an error is larger than 5%, it is counted as failure.
                if (err[j]/float(l)) > 0.05:
                    tempFailures += 1
            listErr[i,0] = temp/(float(len(err))/5)
            listFailures[i,0] = tempFailures/(float(len(err))/5)
            
        print('Avg Detection Error:', listErr)
        print('Failure Rate:',listFailures)
