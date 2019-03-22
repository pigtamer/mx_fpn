import mxnet as mx, cv2 as cv
from mxnet.gluon import nn, loss as gloss, data as gdata
from mxnet import nd, image, autograd
from utils import utils, predata
from utils.utils import calc_loss, cls_eval, bbox_eval
from utils.train import validate
import matplotlib.pyplot as plt
import fpn
import time, argparse
import mxboard as mxb

sw = mxb.SummaryWriter(logdir='./logs', flush_secs=5)

# parsing cli arguments
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load", dest="load",
                    help="bool: load model to directly infer rather than training",
                    type=int, default=1)
parser.add_argument("-b", "--base", dest="base",
                    help="bool: using additional base network",
                    type=int, default=0)
parser.add_argument("-e", "--epoches", dest="num_epoches",
                    help="int: trainig epoches",
                    type=int, default=20)
parser.add_argument("-bs", "--batch_size", dest="batch_size",
                    help="int: batch size for training",
                    type=int, default=4)
parser.add_argument("-is", "--imsize", dest="input_size",
                    help="int: input size",
                    type=int, default=256)
parser.add_argument("-mp", "--model_path", dest="model_path",
                    help="str: the path to load and save model",
                    type=str, default="./FPN-0000.params")
parser.add_argument("-tp", "--test_path", dest="test_path",
                    help="str: the path to your test img",
                    type=str, default="../data/uav/drone_video/Video_233.mp4")
args = parser.parse_args()

ctx = mx.gpu()
net = fpn.ResNet_FPN(num_layers=3)
net.initialize(init="Xavier", ctx=ctx)
net.hybridize()

batch_size, edge_size = args.batch_size, args.input_size
# train_iter, _ = predata.load_data_pikachu(batch_size, edge_size)
train_iter, val_iter = predata.load_data_uav(batch_size, edge_size)
batch = train_iter.next()

# net.initialize(init=init.Xavier(), ctx=ctx)
trainer = mx.gluon.Trainer(net.collect_params(), 'sgd',
                           {'learning_rate': 0.2, 'wd': 5e-4})
cls_loss = gloss.SoftmaxCrossEntropyLoss()
bbox_loss = gloss.L1Loss()

IF_LOAD_MODEL = args.load
if IF_LOAD_MODEL:
    # net.load_parameters(args.model_path) # for save_parameters
    net = nn.SymbolBlock.imports("FPN-symbol.json", ['data'], "FPN-0000.params", ctx=ctx)
    sw.add_graph(net)
else:
    val_recorder = [None] * int(args.num_epoches / 5)
    for epoch in range(args.num_epoches):
        acc_sum, mae_sum, n, m = 0.0, 0.0, 0, 0
        train_iter.reset()  # reset data iterator to read-in images from beginning
        start = time.time()
        for batch in train_iter:
            X = batch.data[0].as_in_context(ctx)
            Y = batch.label[0].as_in_context(ctx)
            with autograd.record():
                # generate anchors and generate bboxes
                anchors, cls_preds, bbox_preds = net(X)
                # print(net)

                # assign classes and bboxes for each anchor
                bbox_labels, bbox_masks, cls_labels = nd.contrib.MultiBoxTarget(anchors, Y,
                                                                                cls_preds.transpose((0, 2, 1)))
                # calc loss
                l = calc_loss(cls_loss, bbox_loss, cls_preds, cls_labels,
                              bbox_preds, bbox_labels, bbox_masks)
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
            _, val_recorder[int(epoch / 5)], _ = validate(val_iter, net, ctx)
            """
            # bs64e100is128
            x128=\
            [0.0044571314102563875, 0.004714686355311359, 0.004102993360805884, 
            0.00472541781135527, 0.004546560210622719, 0.004303313873626369, 
            0.004500057234432253, 0.004181690705128194, 0.004525097298534786, 
            0.004360548305860856, 0.004160227793040261, 0.004485748626373631, 
            0.00415307348901095, 0.004371279761904767, 0.004013564560439553, 
            0.0039706387362636875, 0.0038418612637363125, 0.003791781135531136, 
            0.003516340430402942, 0.003537803342490875]
            # bs64e100is256
            x256=\
            [0.006949512076465214, 0.006133921417124544, 0.0031756167010073,
            0.003042367788461564, 0.0032480540293040594, 0.002543355082417542,
            0.002817007211538436, 0.0027374155792124766, 0.0023591317536629797,
            0.0027544070512820484, 0.002802698603479814, 0.0025201035943223093,
            0.002416366185897467, 0.0022804344093406703, 0.0022929544413919922,
            0.0027705042353479703, 0.0025192093063186594, 0.0023627089056776907,
            0.002537989354395642, 0.0024771777701465547]
            #bs16e100is512
            x512=\
            [0.0029503249154690936, 0.002708454406170735, 0.0028020106896309294, 
            0.0027296421527190917, 0.002767614997182255, 0.0030353510672019857, 
            0.0029239090236686804, 0.0022170087524654436, 0.0024863958157227417, 
            0.002221961732178035, 0.0020293458544660137, 0.002403295822766993, 
            0.002335605100028171, 0.0025928848795435666, 0.0022428743131868156, 
            0.0021350094216681104, 0.0026746090448013238, 0.002247552127359831, 
            0.002425309065934078, 0.002347162052690921]
            """
            print(val_recorder)
    plt.figure()
    plt.plot(val_recorder)
    plt.title("validating curve");
    plt.show()


def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_context(ctx))
    cls_probs = cls_preds.softmax().transpose((0, 2, 1))
    output = nd.contrib.MultiBoxDetection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
    if idx == []: return nd.array([[0, 0, 0, 0, 0, 0, 0]])
    # raise ValueError("NO TARGET. Seq Terminated.")
    return output[0, idx]


def display(img, output, frame_idx=0, threshold=0):
    lscore = []
    for row in output:
        lscore.append(row[1].asscalar())
    for row in output:
        score = row[1].asscalar()
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
        cv.rectangle(img, (bbox[0][0].asscalar(), bbox[0][1].asscalar()),
                     (bbox[0][2].asscalar(), bbox[0][3].asscalar()), (1. * (1 - score), 1. * score, 1. * (1 - score)),
                     int(10 * score))
        if score == max(lscore):
            cv.putText(img, "f%s:%3.2f" % (frame_idx, score), org=(bbox[0][0].asscalar(), bbox[0][1].asscalar()),
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0))
        cv.imshow("res", img)
    cv.waitKey(10)


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
    print("# %d     SPF: %3.2f" % (rd, countt))

    display(frame / 255, output, frame_idx=rd, threshold=0)
    plt.show()
    rd += 1
