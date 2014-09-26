# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>


# requirements
#
#   numpy>=1.6.2
#   scipy>=0.10.1
#   sklearn>=0.12

import optparse
import time

import numpy as np

from cnnrandom import BatchExtractor

from svm import svm_ova_from_splits as svm
from rndsplits import PubFig83
from rndsplits import CalTech256
from models import models
from util import load_imgs

datasets = {'01': PubFig83,
            '02': CalTech256,
           }

def eval(dataset, dataset_path, model):

    in_shape = model['in_shape']
    hps = model['model']

    data = dataset(dataset_path, img_shape=in_shape)
    extractor = BatchExtractor(in_shape=in_shape, model=hps)

    print 'loading images...'
    fnames = data.meta['all_fnames']
    imgs = load_imgs(fnames, out_shape=in_shape, dtype='float32',
                     flatten=len(in_shape)==2, minmax_norm=True)

    if len(imgs) > 0:

        feat_set = extractor.extract(imgs)

        print 'feat_set.shape', feat_set.shape
        feat_set.shape = feat_set.shape[0], -1

        # -- make sure features were properly extracted
        assert(not np.isnan(feat_set).any())
        assert(not np.isinf(feat_set).any())

        print 'evaluating feature according to the dataset protocol...'
        t0 = time.time()
        r_dict = data.protocol_eval(svm, feat_set)

        print r_dict
        print 'protocol executed in %g seconds...' % (time.time() - t0)

    else:
        print 'no images in the given path'

    print 'done!'


def get_optparser():

    usage = "usage: %prog <dataset_path>"

    dataset_options = ''
    for k in sorted(datasets.keys()):
      dataset_options +=  ("     %s - %s \n" % (k, datasets[k].__name__))

    usage = ("usage: %prog <DATASET> <DATASET_PATH>\n\n"
             "DATASET is an integer corresponding to the following supported "
             "datasets:\n" + dataset_options
            )

    parser = optparse.OptionParser(usage=usage)

    model_default = models['default']
    models.pop('default', None)
    model_opts = ' OPTIONS=%s' % (models.keys())

    parser.add_option("--model", "-M",
                      default=model_default,
                      type="str",
                      metavar="STR",
                      help="[DEFAULT='%default']" + model_opts)

    return parser

def main():
    parser = get_optparser()
    opts, args = parser.parse_args()

    if len(args) != 2:
        parser.print_help()
    else:

        try:
            dataset = datasets[args[0]]
        except KeyError:
            raise ValueError('invalid dataset option')

        dataset_path = args[1]

        try:
            model = models[opts.model]
            assert type(model) == dict
        except KeyError:
            raise ValueError('invalid model')

        eval(dataset, dataset_path, model)

if __name__ == "__main__":
    main()
