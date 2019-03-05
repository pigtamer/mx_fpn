import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet import autograd, init, contrib, nd, sym

cls_lossfunc = gloss.SoftmaxCrossEntropyLoss()

bbox_lossfunc = gloss.L1Loss()





def training(data_iter, num_epoches, cls_lossfunc, bbox_lossfunc):
    # TODO: define the way that the model should be trained
    #   wth gluon.Trainer(...)
    for eph in range(num_epoches):
        pass
    pass



