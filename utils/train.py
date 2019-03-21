import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet import autograd, init, contrib, nd, sym
from utils.utils import calc_loss, cls_eval, bbox_eval

cls_lossfunc = gloss.SoftmaxCrossEntropyLoss()

bbox_lossfunc = gloss.L1Loss()


def training(data_iter, num_epoches, cls_lossfunc, bbox_lossfunc):
    # TODO: define the way that the model should be trained
    #   wth gluon.Trainer(...)
    for eph in range(num_epoches):
        pass
    pass


def validate(val_iter, net, ctx=mx.gpu()):
    for batch in val_iter:
        X = batch.data[0].as_in_context(ctx)
        Y = batch.label[0].as_in_context(ctx)
        # generate anchors and generate bboxes
        anchors, cls_preds, bbox_preds = net(X)
        # assign classes and bboxes for each anchor
        bbox_labels, bbox_masks, cls_labels = nd.contrib.MultiBoxTarget(anchors, Y,
                                                                        cls_preds.transpose((0, 2, 1)))
        # calc loss
        l = calc_loss(cls_lossfunc, bbox_lossfunc, cls_preds, cls_labels,
                      bbox_preds, bbox_labels, bbox_masks)
        acc_cls = cls_eval(cls_preds, cls_labels)
        acc_bbox = bbox_eval(bbox_preds, bbox_labels, bbox_masks)

    return (l, acc_cls, acc_bbox)
