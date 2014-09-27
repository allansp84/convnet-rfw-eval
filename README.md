## convnet-rfw-eval

This repository aggregates pieces of code required to evaluate the performance of features extracted with [convnet-rfw](http:/github.com/giovanichiachia/convnet-rfw) in several public datasets.

The models provided in `models.py` were optimized with [hyperopt](https://github.com/hyperopt/hyperopt), and details to obtain them yourself will follow shortly.

Supported datasets:

* [PubFig83](https://www.dropbox.com/s/0ez5p9bpjxobrfv/pubfig83-aligned.tar.bz2)
* [CalTech256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar)

While the main requirement is [convnet-rfw](http:/github.com/giovanichiachia/convnet-rfw), other dependencies to execute the code are:

* numpy>=1.6.1
* scipy>=0.10.1
* joblib>=0.6.4
* [scikit-learn](http://scikit-learn.org/)>=0.12

Make sure they are all available in your python environment. In order to avoid disk swapping, evaluations on the supported datasets require 12GB of RAM.

Once everything is setup properly, you should be able to get something like this:

```
python eval.py 01 <PUBFIG83_PATH> -M pubfig83
...
feat_set.shape (13838, 7, 7, 256)
evaluating feature according to the dataset protocol...
{'acc': 0.8943373, 'loss': 0.10566270351409912}
protocol executed in 383.107 seconds...
```

```
python eval.py 02 <CALTECH256_PATH> -M caltech256
...
feat_set.shape (30607, 10, 10, 256)
evaluating feature according to the dataset protocol...
{'acc': 0.22682288, 'loss': 0.77317711710929871}
protocol executed in 1574.94 seconds...
```