import mxnet as mx
from mxnet.gluon import nn, contrib
from mxnet import nd, sym
from ssd.anchor_params import *
from utils.utils import concat_preds
from syms import resnet50_hybrid as res50h
import ssd


class ChannelAdapt(nn.Block):
    def __init__(self, out_channels, **kwargs):
        super(ChannelAdapt, self).__init__()
        self.net = nn.Sequential()
        self.net.add(
            nn.Conv2D(channels=out_channels, kernel_size=(1, 1)),
            nn.BatchNorm(in_channels=out_channels),
            nn.Activation(activation='relu')
        )

    def forward(self, x, *args, **kwargs):
        return self.net(x)


def getChannelAdapt(out_channels):
    net = nn.Sequential()
    net.add(
        nn.Conv2D(channels=out_channels, kernel_size=(1, 1)),
        nn.BatchNorm(in_channels=out_channels),
        nn.Activation(activation='relu')
    )
    return net


def fusionFMaps(lMap, sMap, method='upconv'):
    if method == 'upconv':
        raise Exception("NOT IMPLEMENTED YET.")
        # upconver = nd.Deconvolution(data=sMap, kernel=upconv_ksize,
        #                             num_filter=512, stride=(1, 1), weight=nd.random.normal(0, 1, (512, 512, 3, 3)))
        # upconver = nd.Activation(data=upconver, act_type='relu')
        # upconv_sMap = nd.BatchNorm(data=upconver, gamma=nd.random_gamma(alpha=9, beta=0.5, shape=(2, 2)))
        # # upconver.initialize(ctx=mx.gpu())  # how to init? should I make the params trainable?
        # upconv_sMap = nd.contrib.BilinearResize2D(data=sMap,
        #                                           height=lMap.shape[-2], width=lMap.shape[-1])
    elif method == 'bilinear':
        upconv_sMap = nd.contrib.BilinearResize2D(data=sMap,
                                                  height=lMap.shape[-2], width=lMap.shape[-1])

    else:
        raise Exception("ERROR! [jcy checkpoint]: Unexpected enlarging method.")

    res = (lMap + upconv_sMap) / 2 # add large fmap with the smaller one

    # res = nd.broadcast_div(res, nd.max(res))
    return res


class VGG_FPN(nn.Block):
    def __init__(self, num_layers=3, num_classes=1, **kwargs):
        super().__init__(**kwargs)
        self.BaseBlk = ssd.BaseNetwork(IF_TINY=True)
        self.feature_blk_1 = nn.Sequential()
        self.feature_blk_1.add(
            nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(channels=1024, kernel_size=3, padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=(2, 2))
        )
        self.feature_blk_2 = nn.Sequential()
        self.feature_blk_2.add(
            nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=(2, 2))
        )
        self.ssd_1 = ssd.LightRetina(num_cls=1, num_ach=retina_num_anchors)
        self.ssd_2 = ssd.LightRetina(num_cls=1, num_ach=retina_num_anchors)
        self.ssd_3 = ssd.LightRetina(num_cls=1, num_ach=retina_num_anchors)

        self.chan_align_32 = getChannelAdapt(out_channels=1024)
        self.chan_align_21 = getChannelAdapt(out_channels=256)

    def forward(self, x):
        x = self.BaseBlk(x)
        fmap_1 = x
        fmap_2 = self.feature_blk_1(fmap_1)
        fmap_3 = self.feature_blk_2(fmap_2)

        fusion_33 = fmap_3  # placeholder. to be deleted in the future
        fusion_32 = fusionFMaps(fmap_2, self.chan_align_32(fusion_33),
                                method='bilinear')
        fusion_21 = fusionFMaps(fmap_1, self.chan_align_21(fusion_32),
                                method='bilinear')

        anchors, cls_preds, bbox_preds = [None] * 3, [None] * 3, [None] * 3
        anchors[2], cls_preds[2], bbox_preds[2] = self.ssd_3(fusion_33)
        anchors[1], cls_preds[1], bbox_preds[1] = self.ssd_2(fusion_32)
        anchors[0], cls_preds[0], bbox_preds[0] = self.ssd_1(fusion_21)

        # print("-----------------------------------------------\n"
        #       "FPN:     [top -> bottom]\n"
        #       "         fusion[3]: %s;\n"
        #       "         fusion[2]: %s;\n"
        #       "         fusion[1]: %s;\n"
        #       "         input    : %s;\n"
        #       "-----------------------------------------------\n"
        #       % (fusion_33.shape, fusion_32.shape, fusion_21.shape, x.shape))
        return (nd.concat(*anchors, dim=1),
                nd.concat(*cls_preds, dim=1),
                nd.concat(*bbox_preds, dim=1))


class ResNet_FPN(nn.Block):
    def __init__(self, num_layers=3, num_classes=1, **kwargs):
        super().__init__(**kwargs)
        self.BaseBlk = res50h.ResNet50(
            params={
                "channels":
                    [[[64, 64, 256]] * 3,
                     [[128, 128, 512]] * 4],  # two residual layers.
                # each having 3 or 4 units
                "kernel_sizes":
                    [[[1, 3, 1]] * 3,
                     [[1, 3, 1]] * 4],
                "branches":
                    [{"channel": 256, "kernel_size": 1, "stride": 1, "padding": 0},
                     {"channel": 512, "kernel_size": 1, "stride": 2, "padding": 0}]
            },
            IF_DENSE=False,
            IF_HEAD=True
        )
        self.feature_blk_1 = res50h.ResNet50(
            params={"channels": [[[256, 256, 1024]] * 6],
                    "kernel_sizes": [[[1, 3, 1]] * 6],
                    "branches":
                        [{"channel": 1024, "kernel_size": 1, "stride": 2, "padding": 0}]
                    },
            IF_DENSE=False,
            IF_HEAD=True
        )
        self.feature_blk_2 = res50h.ResNet50(
            params={"channels": [[[512, 512, 2048]] * 3],
                    "kernel_sizes": [[[1, 3, 1]] * 3],
                    "branches":
                        [{"channel": 2048, "kernel_size": 1, "stride": 2, "padding": 0}]
                    },
            IF_DENSE=False,
            IF_HEAD=True
        )

        self.ssd_1 = ssd.LightRetina(num_cls=1, num_ach=retina_num_anchors)
        self.ssd_2 = ssd.LightRetina(num_cls=1, num_ach=retina_num_anchors)
        self.ssd_3 = ssd.LightRetina(num_cls=1, num_ach=retina_num_anchors)

        self.chan_align_32 = getChannelAdapt(out_channels=1024)
        self.chan_align_21 = getChannelAdapt(out_channels=512)

    def forward(self, x):
        x = self.BaseBlk(x)
        fmap_1 = x
        fmap_2 = self.feature_blk_1(fmap_1)
        fmap_3 = self.feature_blk_2(fmap_2)

        fusion_33 = fmap_3  # placeholder. to be deleted in the future
        fusion_32 = fusionFMaps(fmap_2, self.chan_align_32(fusion_33),
                                method='bilinear')
        fusion_21 = fusionFMaps(fmap_1, self.chan_align_21(fusion_32),
                                method='bilinear')

        anchors, cls_preds, bbox_preds = [None] * 3, [None] * 3, [None] * 3
        anchors[2], cls_preds[2], bbox_preds[2] = self.ssd_3(fusion_33)
        anchors[1], cls_preds[1], bbox_preds[1] = self.ssd_2(fusion_32)
        anchors[0], cls_preds[0], bbox_preds[0] = self.ssd_1(fusion_21)

        # print("-----------------------------------------------\n"
        #       "FPN:     [top -> bottom]\n"
        #       "         fusion[3]: %s;\n"
        #       "         fusion[2]: %s;\n"
        #       "         fusion[1]: %s;\n"
        #       "         input    : %s;\n"
        #       "-----------------------------------------------\n"
        #       % (fusion_33.shape, fusion_32.shape, fusion_21.shape, x.shape))
        return (nd.concat(*anchors, dim=1),
                nd.concat(*cls_preds, dim=1),
                nd.concat(*bbox_preds, dim=1))


def test():
    # from mxnet import image
    # xorig = image.imread("/home/cunyuan/code/img/berry.jpg")
    # print(xorig.shape)
    # x = xorig.transpose((2, 0, 1)).expand_dims(0).astype('float32')
    # print(x.shape)
    # x1 = image.imresize(xorig, 64, 64).transpose((2, 0, 1)).\
    #     expand_dims(0).astype('float32')
    # print(x1.shape)
    #
    # fusionFMaps(x, x1)
    import time
    x = nd.random.normal(0, 1, (1, 100, 512, 512), ctx=mx.gpu())
    net = ResNet_FPN()
    net.hybridize()
    net.initialize(ctx=mx.gpu())
    cnt = time.time()
    res = net(x)
    print(time.time() - cnt)
    net.export("./res_fpn")
    print(res)

# test()
