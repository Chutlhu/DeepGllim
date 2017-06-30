# Deep Gllim
Perform deep regression by training jointly a CNN (feature extraction) and a Gllim model (inverse regression).
![EM](https://team.inria.fr/perception/files/2017/04/EM.png)

## Installation
`python setup.py install`

## Requirements
- Numpy, Scipy
- Scikit-learn
- Theano
- Keras

## Data
The `train.txt` and `test.txt` files contain, line by line, the path to the image followed by the values of the regressors, like this:
```
path/to/im1.jpg y1 y2 y3
path/to/im2.jpg y1 y2 y3
...
```

## Run
`THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python deep_gllim.py train.txt test.txt`

## Models
You can choose between:
- VGG16
- VGG19

## Cite

```
@inproceedings{lathuiliere:hal-01504847,
  TITLE = {Deep Mixture of Linear Inverse Regressions Applied to Head-Pose Estimation},
  AUTHOR = {Lathuili{\`e}re, St{\'e}phane and Juge, R{\'e}mi and Mesejo, Pablo and Mu{\~n}oz-Salinas, Rafael and Horaud, Radu},
  BOOKTITLE = {{IEEE Conference on Computer Vision and Pattern Recognition}},
  ADDRESS = {Honolulu, Hawaii, United States},
  ORGANIZATION = {{IEEE Computer Society}},
  YEAR = {2017},
  MONTH = Jul,
  PDF = {https://hal.inria.fr/hal-01504847/file/main.pdf},
}
```