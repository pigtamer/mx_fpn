import mxnet as mx, cv2 as cv
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
                    type=int, default=1)
parser.add_argument("-b", "--base", dest="base",
                    help="bool: using additional base network",
                    type=int, default=0)
parser.add_argument("-bs", "--batch_size", dest="batch_size",
                    help="int: batch size for training",
                    type=int, default=0)
parser.add_argument("-s", "--imsize", dest="input_size",
                    help="int: input size",
                    type=int, default=448)
parser.add_argument("-m", "--model_path", dest="model_path",
                    help="str: the path to load and save model",
                    type=str, default="./FPN-0000.params")
parser.add_argument("-t", "--test_path", dest="test_path",
                    help="str: the path to your test img",
                    type=str, default="../data/uav/drone_video/Video_233.mp4")
args = parser.parse_args()

ctx = mx.gpu()
net = fpn.ResNet_FPN(num_layers=3)
net.initialize(init="Xavier", ctx=ctx)
net.hybridize()

batch_size, edge_size = args.batch_size, args.input_size
# train_iter, _ = predata.load_data_pikachu(batch_size, edge_size)
train_iter, _ = predata.load_data_uav(batch_size, edge_size)
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
    # net.load_parameters(args.model_path) # for save_parameters
    net = nn.SymbolBlock.imports("FPN-symbol.json", ['data'], "FPN-0000.params", ctx=ctx)

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
                #print(net)

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
            print('epoch %2d, class err %.2e, bbox mae %.2e, time %.1f sec' % (
                epoch + 1, 1 - acc_sum / n, mae_sum / m, time.time() - start))

            # Checkpoint
            if (epoch + 1) % 5 == 0:
                net.export('FPN')


# img = image.imread(args.test_path)
# feature = image.imresize(img, args.input_size, args.input_size).astype('float32')
# X = feature.transpose((2, 0, 1)).expand_dims(axis=0)


def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_context(ctx))
    cls_probs = cls_preds.softmax().transpose((0, 2, 1))
    output = nd.contrib.MultiBoxDetection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
    if idx == []: return nd.array([[0, 0, 0, 0, 0, 0, 0]])
    # raise ValueError("NO TARGET. Seq Terminated.")
    return output[0, idx]


def display(img, output, threshold):
    lscore = []
    for row in output:
        lscore.append(row[1].asscalar())
    for row in output:
        score = row[1].asscalar()
        if score < min(max(lscore), threshold):
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
        cv.rectangle(img, (bbox[0][0].asscalar(), bbox[0][1].asscalar()),
                     (bbox[0][2].asscalar(), bbox[0][3].asscalar()), (0, 255, 0), 3)
        cv.imshow("res", img)
        cv.waitKey(60)


cap = cv.VideoCapture(args.test_path)
rd = 0
while True:
    ret, frame = cap.read()
    img = nd.array(frame)
    feature = image.imresize(img, 256, 256).astype('float32')
    X = feature.transpose((2, 0, 1)).expand_dims(axis=0)

    countt = time.time()
    output = predict(X)
    # if rd == 0: net.export('ssd')
    countt = time.time() - countt
    print("SPF: %3.2f" % countt)

    utils.set_figsize((5, 5))

    display(frame / 255, output, threshold=0.8)
    plt.show()
    rd += 1
