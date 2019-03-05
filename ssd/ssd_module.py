import cv2 as cv
import mxnet as mx
from utils.utils import concat_preds
from ssd import anchor_params
from mxnet.gluon import nn, loss as gloss, data as gdata
from mxnet.ndarray import contrib
from mxnet import nd, image

""""
def gen_ClsNet(K, A):
    clsNet = nn.Sequential()
    for k in range(4):
        clsNet.add(
            nn.Conv2D(channels=256, kernel_size=1, padding=0)
        )
    clsNet.add(nn.Conv2D(channels=int(K*A), kernel_size=1, padding=0))
    return clsNet


def gen_BBoxNet(A):
    bboxNet = nn.Sequential()
    for k in range(4):
        bboxNet.add(
            nn.Conv2D(channels=256, kernel_size=1, padding=0)
        )
    bboxNet.add(nn.Conv2D(channels=int(4*A), kernel_size=1, padding=0))
    return bboxNet


class PredBranch(nn.Block):
    def __init__(self, K, A, **kwargs):
        super(PredBranch, self).__init__(**kwargs) ## inherit
        self.clsBlk = gen_ClsNet(K, A)
        self.bboxBlk = gen_BBoxNet(A)
    def forward(self, fmap):
        # anchors, cls_preds, bbox_preds = contrib.nd.MultiBoxTarget()???
        pass

"""

def genClsPredictor(num_cls, num_ach):
    return nn.Conv2D(num_ach * (num_cls + 1), kernel_size=3, padding=1)


def genBBoxRegressor(num_ach):
    return nn.Conv2D(num_ach * 4, kernel_size=3, padding=1)


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = contrib.MultiBoxPrior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

class LightSSD(nn.Block):
    def __init__(self, num_cls, num_ach, IF_TINY=True, **kwargs):
        super(LightSSD, self).__init__(**kwargs)
        self.num_classes = num_cls
        self._IF_TINY = IF_TINY
        if not self._IF_TINY:
            self.BaseBlk = BaseNetwork(False)
        self.blk1 = nn.Sequential()
        self.blk1.add(nn.Conv2D(channels=1024, kernel_size=3, strides=1, padding=0),
                      nn.Conv2D(channels=1024, kernel_size=1, strides=1, padding=1),
                      nn.BatchNorm(in_channels=1024),
                      nn.Activation('relu'),
                      nn.MaxPool2D(2))

        self.cls1 = genClsPredictor(num_cls, num_ach)
        self.reg1 = genBBoxRegressor(num_ach)

        self.blk2 = nn.Sequential()
        self.blk2.add(nn.Conv2D(channels=256, kernel_size=1, strides=1, padding=0),
                      nn.Conv2D(channels=512, kernel_size=3, strides=1, padding=1),
                      nn.BatchNorm(in_channels=512),
                      nn.Activation('relu'),
                      nn.MaxPool2D(2))

        self.cls2 = genClsPredictor(num_cls, num_ach)
        self.reg2 = genBBoxRegressor(num_ach)

        self.blk3 = nn.Sequential()
        self.blk3.add(nn.Conv2D(channels=128, kernel_size=1, strides=1, padding=0),
                      nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1),
                      nn.BatchNorm(in_channels=256),
                      nn.Activation('relu'),
                      nn.MaxPool2D(2))

        self.cls3 = genClsPredictor(num_cls, num_ach)
        self.reg3 = genBBoxRegressor(num_ach)

        self.blk4 = nn.Sequential()
        self.blk4.add(nn.Conv2D(channels=128, kernel_size=1, strides=1, padding=0),
                      nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1),
                      nn.BatchNorm(in_channels=256),
                      nn.Activation('relu'),
                      nn.MaxPool2D(2))

        self.cls4 = genClsPredictor(num_cls, num_ach)
        self.reg4 = genBBoxRegressor(num_ach)

        self.blk5 = nn.Sequential()
        self.blk5.add(nn.Conv2D(channels=128, kernel_size=1, strides=1, padding=0),
                      nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1),
                      nn.BatchNorm(in_channels=256),
                      nn.Activation('relu'),
                      nn.MaxPool2D(2))

        self.cls5 = genClsPredictor(num_cls, num_ach)
        self.reg5 = genBBoxRegressor(num_ach)

    def forward(self, x):
        if not self._IF_TINY:
            x = self.BaseBlk(x)
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for k in range(5):
            (x, anchors[k], cls_preds[k], bbox_preds[k]) = \
                blk_forward(x, getattr(self, "blk%d" % (k + 1)),
                            anchor_params.sizes[k], anchor_params.ratios[k],
                            getattr(self, "cls%d" % (k + 1)), getattr(self, "reg%d" % (k + 1)))
            print("SSD:     layer[%d], fmap shape %s, anchor %s" % (k + 1, x.shape, anchors[k].shape))
        return (nd.concat(*anchors, dim=1),
                concat_preds(cls_preds).reshape((0, -1, self.num_classes + 1)),
                concat_preds(bbox_preds))


class BaseNetwork(nn.Block):  # VGG base network, without fc
    def __init__(self, IF_TINY, **kwargs):
        super(BaseNetwork, self).__init__(**kwargs)
        self.IF_TINY = IF_TINY
        self.conv1_1 = nn.Conv2D(channels=64, kernel_size=3, padding=1, activation='relu')
        self.conv1_2 = nn.Conv2D(channels=64, kernel_size=3, padding=1, activation='relu')
        self.pool1 = nn.MaxPool2D(pool_size=(2, 2))
        self.conv2_1 = nn.Conv2D(channels=128, kernel_size=3, padding=1, activation='relu')
        self.conv2_2 = nn.Conv2D(channels=128, kernel_size=3, padding=1, activation='relu')
        self.pool2 = nn.MaxPool2D(pool_size=(2, 2))
        self.conv3_1 = nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu')
        self.conv3_2 = nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu')
        self.conv3_3 = nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu')
        self.pool3 = nn.MaxPool2D(pool_size=(2, 2))  # smaller here
        if not self.IF_TINY:
            self.conv4_1 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.conv4_2 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.conv4_3 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.pool4 = nn.MaxPool2D(pool_size=(2, 2))
            self.conv5_1 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.conv5_2 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.conv5_3 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.pool5 = nn.MaxPool2D(pool_size=(2, 2))

    def forward(self, x):
        x = self.pool1(self.conv1_2(self.conv1_1(x)))
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        x = self.pool3(self.conv3_3(self.conv3_2(self.conv3_1(x))))
        if not self.IF_TINY:
            x = self.pool4(self.conv4_3(self.conv4_2(self.conv4_1(x))))
            x = self.pool5(self.conv5_3(self.conv5_2(self.conv5_1(x))))
        return x