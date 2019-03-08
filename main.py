import mxnet as mx
from mxnet.gluon import nn, loss as gloss, data as gdata
from mxnet import nd, image, autograd
from utils import utils, predata
import matplotlib.pyplot as plt
import fpn
import time, argparse

# parsing cli arguments
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load", dest="load",
                    help="bool: load model to directly infer rather than training",
                    type=bool, default=True)
parser.add_argument("-b", "--base", dest="base",
                    help="bool: using additional base network",
                    type=bool, default=False)
parser.add_argument("-s", "--imsize", dest="input_size",
                    help="int: input size",
                    type=int, default=448)
parser.add_argument("-m", "--model_path", dest="model_path",
                    help="str: the path to load and save model",
                    type=str, default="./myfpn.params")
parser.add_argument("-t", "--test_path", dest="test_path",
                    help="str: the path to your test img",
                    type=str, default="./img/pikachu_ms.jpg")
args = parser.parse_args()

ctx = mx.gpu()
net = fpn.FPN(num_layers=3)
net.initialize(init="Xavier", ctx=ctx)

batch_size, edge_size = 1, args.input_size
train_iter, _ = predata.load_data_pikachu(batch_size, edge_size)
batch = train_iter.next()

if batch_size == 25:  # show fucking pikachuus in grid
    imgs = (batch.data[0][0:25].transpose((0, 2, 3, 1))) / 255
    axes = utils.show_images(imgs, 5, 5).flatten()
    for ax, label in zip(axes, batch.label[0][0:25]):
        utils.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])

    plt.show()

# net.initialize(init=init.Xavier(), ctx=ctx)
trainer = mx.gluon.Trainer(net.collect_params(), 'sgd',
                           {'learning_rate': 0.2, 'wd': 5e-4})
cls_loss = gloss.SoftmaxCrossEntropyLoss()
bbox_loss = gloss.L1Loss()


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    # the result from class prediction is at the last dim
    # argmax() should be assigned with the last dim of cls_preds
    return (cls_preds.argmax(axis=-1) == cls_labels).sum().asscalar()


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return ((bbox_labels - bbox_preds) * bbox_masks).abs().sum().asscalar()


IF_LOAD_MODEL = args.load
if IF_LOAD_MODEL:
    net.load_parameters(args.model_path)
else:
    for epoch in range(20):
        acc_sum, mae_sum, n, m = 0.0, 0.0, 0, 0
        train_iter.reset()  # reset data iterator to read-in images from beginning
        start = time.time()
        for batch in train_iter:
            X = batch.data[0].as_in_context(ctx)
            Y = batch.label[0].as_in_context(ctx)
            with autograd.record():
                # generate anchors and generate bboxes
                anchors, cls_preds, bbox_preds = net(X)
                print(net)
                # assign classes and bboxes for each anchor
                bbox_labels, bbox_masks, cls_labels = nd.contrib.MultiBoxTarget(anchors, Y,
                                                                                cls_preds.transpose((0, 2, 1)))
                # calc loss
                l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                              bbox_masks)
            l.backward()
            trainer.step(batch_size)
            acc_sum += cls_eval(cls_preds, cls_labels)
            n += cls_labels.size
            mae_sum += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
            m += bbox_labels.size

        if (epoch + 1) % 1 == 0:
            print('epoch %2d, class err %.2e, bbox mae %.2e, time %.1f sec' % (
                epoch + 1, 1 - acc_sum / n, mae_sum / m, time.time() - start))
    net.save_parameters(args.model_path)

img = image.imread(args.test_path)
feature = image.imresize(img, args.input_size, args.input_size).astype('float32')
X = feature.transpose((2, 0, 1)).expand_dims(axis=0)


def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_context(ctx))
    cls_probs = cls_preds.softmax().transpose((0, 2, 1))
    output = nd.contrib.MultiBoxDetection(cls_probs, bbox_preds, anchors, nms_threshold=0.5)
    idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
    if idx == []:
        raise ValueError("NO TARGET. Seq Terminated.")
    return output[0, idx]


countt = time.time()
output = predict(X)
countt = time.time() - countt
print("SPF: %3.2f" % countt)

utils.set_figsize((5, 5))


def display(img, output, threshold):
    fig = utils.plt.imshow(img.asnumpy())
    for row in output:
        score = row[1].asscalar()
        if score < threshold[0] or score > threshold[1]:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
        utils.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')


display(img, output, threshold=(0.8, 1))
plt.show()




