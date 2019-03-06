import mxnet as mx
from mxnet.gluon import nn, contrib
from mxnet import nd
from ssd.anchor_params import *
from utils.utils import concat_preds
import ssd

def fusionFMaps(lMap, sMap, upconv_ksize=3, method='upconv'):
    # lMap/sMap stand for large/small feature maps
    # methods: 'upconv', 'lin_interpol'
    s_channels = sMap.shape[1]
    l_channels = lMap.shape[1]
    if s_channels != l_channels:
        raise ValueError("ERROR [jcy checkpoint]: Inconsistent feature-map channels."
                         " Check the channels of neighboring layers. ")
    if method == 'upconv':
        upconver = nn.Sequential()
        upconver.add(nn.Conv2DTranspose(channels=l_channels, kernel_size=upconv_ksize,
                                        activation='relu'),
                     nn.BatchNorm(in_channels=l_channels))
        upconver.initialize(ctx=mx.gpu())  # how to init? should I make the params trainable?
        upconv_sMap = upconver(sMap)
        # TODO: Modify this. Figure out a way to deal with size problem brought by pooling
        upconv_sMap = nd.contrib.BilinearResize2D(
            data=upconv_sMap, height=lMap.shape[-2], width=lMap.shape[-1])
    elif method == 'bilinear':
        upconv_sMap = nd.contrib.BilinearResize2D(
            data=sMap, height=lMap.shape[-2], width=lMap.shape[-1])
        # NO !! We must unify the feature channels of the up-down path! Or the color would be eliminated.
        # consider re-enable this when things are done and you are ready for the training of
        # the params in upconv blks
        # ^
        # ^ this is not a problem asshole. Do you think that color images are special?
        chan_adapter = nn.Sequential()
        _ = nn.Conv2D(channels=l_channels, kernel_size=1, in_channels=s_channels)
        _.initialize()
        _.weight.set_data(nd.ones((l_channels, s_channels, 1, 1)) / s_channels)
        chan_adapter.add(_,
                         nn.BatchNorm(in_channels=l_channels))
        chan_adapter.initialize()
        upconv_sMap = chan_adapter(upconv_sMap)
    else:
        raise Exception("ERROR! [jcy checkpoint]: Unexpected enlarging method.")

    res = nd.add(lMap, upconv_sMap) / 2  # add large fmap with the smaller one
    res = res / nd.max(res)
    # return (res, upconv_sMap)
    return res

class FPN(nn.Block):
    def __init__(self, num_layers=3, num_classes=1, **kwargs):
        super().__init__(**kwargs)
        #
        #      /=/    ----------o-----> {pred}----------------------
        #       ^               |                                   |
        #      [&]             [>]                                  |
        #       ^               |                                   |
        #    /=====/  -------> (+) ---> /+++++/ --> {pred} ------{concat}   
        #       ^               |                                   |
        #      [&]             [>]                                  |
        #       ^               |                                   |
        # /=============/ ---> (+) ---> /+++++++++++/ --> {pred}-----
        #       ^
        #      [&]
        #       ^
        #       ^
        #  --{input}--
        #
        # 1 -> 3 : bottom -> top
        # self.BaseBlk = ssd.BaseNetwork(True)
        self.feature_blk_1 = nn.Sequential()
        self.feature_blk_1.add(nn.Conv2D(channels=256, kernel_size=3, padding=1),
                               nn.Conv2D(channels=256, kernel_size=3, padding=1),
                               nn.Conv2D(channels=256, kernel_size=3, padding=1),
                               nn.Activation('relu'),
                               nn.MaxPool2D(2),
                               nn.BatchNorm(in_channels=256))
        self.ssd_1 = ssd.LightSSD(num_cls=1, num_ach=num_anchors)

        self.feature_blk_2 = nn.Sequential()
        self.feature_blk_2.add(nn.Conv2D(channels=512, kernel_size=3, padding=1),
                               nn.Conv2D(channels=512, kernel_size=3, padding=1),
                               nn.Conv2D(channels=512, kernel_size=3, padding=1),
                               nn.Activation('relu'),
                               nn.MaxPool2D(2),
                               nn.BatchNorm(in_channels=512))

        self.ssd_2 = ssd.LightSSD(num_cls=1, num_ach=num_anchors)

        self.feature_blk_3 = nn.Sequential()
        self.feature_blk_3.add(nn.Conv2D(channels=512, kernel_size=3, padding=1),
                               nn.Conv2D(channels=512, kernel_size=3, padding=1),
                               nn.Conv2D(channels=512, kernel_size=3, padding=1),
                               nn.Activation('relu'),
                               nn.MaxPool2D(2),
                               nn.BatchNorm(in_channels=512))
        self.ssd_3 = ssd.LightSSD(num_cls=1, num_ach=num_anchors)

    def forward(self, x):
        # x = self.BaseBlk(x)
        fmap_1 = self.feature_blk_1(x)
        fmap_2 = self.feature_blk_2(fmap_1)
        fmap_3 = self.feature_blk_3(fmap_2)

        fusion_33 = fmap_3  # placeholder. to be deleted in the future
        fusion_32 = fusionFMaps(fmap_2, fusion_33, method='upconv')
        fusion_21 = fusionFMaps(fmap_1, fusion_32, method='upconv')

        anchors, cls_preds, bbox_preds = [None] * 3, [None] * 3, [None] * 3
        anchors[2], cls_preds[2], bbox_preds[2] = self.ssd_3(fusion_33)
        anchors[1], cls_preds[1], bbox_preds[1] = self.ssd_2(fusion_32)
        anchors[0], cls_preds[0], bbox_preds[0] = self.ssd_1(fusion_21)

        print("----------\n"
              "FPN:     [top -> bottom]\n"
              "         fusion[3]: %s;\n "
              "         fusion[2]: %s;\n "
              "         fusion[1]: %s"
              %(fusion_33.shape, fusion_32.shape, fusion_21.shape))
        return (nd.concat(*anchors, dim=1),
                nd.concat(*cls_preds, dim=1),
                nd.concat(*bbox_preds, dim=1))

# def test():
#     from mxnet import image
#     xorig = image.imread("/home/cunyuan/code/img/berry.jpg")
#     print(xorig.shape)
#     x = xorig.transpose((2, 0, 1)).expand_dims(0).astype('float32')
#     print(x.shape)
#     x1 = image.imresize(xorig, 64, 64).transpose((2, 0, 1)).\
#         expand_dims(0).astype('float32')
#     print(x1.shape)
#
#
#
#     fusionFMaps(x, x1)
#
#
# test()
