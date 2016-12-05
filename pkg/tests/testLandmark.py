import numpy as np
from test import run_eval
#gt = np.random.randint(1,10,(3, 10))
output = np.array([[5,0 ,5,0 ,5 ,0 ,5 ,0 , 5,5 ]])#,[100, 100, 100, 100, 100, 100, 100, 100, 100, 100], [50, 50, 50,50,50,50,50,50,50,50]])
gt = np.zeros((1,10))

print gt
print output

l=100

run_eval(gt,output,l,'landmark')
