DEEP GLLIM
========================

To run it :

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python deep_gllim.py train.txt test.txt

To run it on the perception cluster:

#!/bin/bash
#OAR -l /host=1/gpudevice=1,walltime=48:01:00
#OAR -O /services/scratch/perception/rjuge/log/scriptCluster_%jobid%.output
#OAR -E /services/scratch/perception/rjuge/log/scriptCluster.error

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python
deep_gllim.py trainingAnnotations.txt testAnnotations.txt

GLLIM model implemented in python from : arXiv:1308.2302v3


