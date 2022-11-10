import argparse

import tensorflow as tf
from tensorflow.keras.optimizers import SGD

from nets.ssd import SSD300
from nets.ssd_training import MultiboxLoss
from utils.anchors import get_anchors
from utils.dataloader import SSDDatasets
from utils.utils import get_classes


class SSD_Train(object):
    def __init__(self):
        # 获取类别以及类别个数
        self.class_names, self.num_classes = get_classes(opt.classes)
        self.num_classes += 1
        self.anchors = get_anchors(opt.input_shape, opt.anchors_size)

    def load_weights(self):
        model = SSD300((opt.input_shape[0], opt.input_shape[1], 3), self.num_classes,
                       weight_decay=opt.weight_decay)
        model.load_weights(opt.weights, by_name=True, skip_mismatch=True)
        return model

    def get_dataloader(self):
        with open(opt.train_annotation_path, encoding='utf-8') as f:
            train_lines = f.readlines()
        with open(opt.val_annotation_path, encoding='utf-8') as f:
            val_lines = f.readlines()

        train_dataloader = SSDDatasets(train_lines, opt.input_shape, self.anchors, opt.batch_size, self.num_classes,
                                       train=True)
        val_dataloader = SSDDatasets(val_lines, opt.input_shape, self.anchors, opt.batch_size, self.num_classes,
                                     train=False)
        return train_dataloader, val_dataloader

    def train(self, model, train_dataloader, val_dataloader):
        optimizer = SGD(learning_rate=opt.learning_rate)
        multiloss = MultiboxLoss(self.num_classes, neg_pos_ratio=3.0).compute_loss
        model.compile(optimizer=optimizer,
                      loss=multiloss,
                      metrics=['accuracy'])

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath="weights/ep{epoch:03d}-acc{acc:.3f}-val_acc{val_acc:.3f}.h5",
            monitor='val_accuracy',
            save_weights_only=True,
            save_best_only=True,
            mode="auto",
            period=5)

        model.fit(
            x=train_dataloader,
            validation_data=val_dataloader,
            epochs=opt.epochs,
            callbacks=[checkpoint]
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="weights/ssd_weights.h5", help="是否需要加载模型训练，模型保存位置/空")
    parser.add_argument('--classes', default="classes/classes.txt", help="预测标签类别")
    parser.add_argument('--epochs', default=50, help="训练轮次")
    parser.add_argument('--input_shape', default=[300, 300], help="训练图像的大小，传入列表")
    parser.add_argument('--anchors_size', default=[21, 45, 99, 153, 207, 261, 315], help="指定先验框的大小")
    parser.add_argument('--batch_size', default=8, help="样本批次")
    parser.add_argument('--weight_decay', default=5e-4, help="权值衰减，可防止过拟合,使用adam时建议设置为0")
    parser.add_argument('--train_annotation_path', default='2007_train.txt', help="训练图片路径和标签")
    parser.add_argument('--val_annotation_path', default='2007_val.txt', help="验证图片路径和标签")
    parser.add_argument('--learning_rate', default=0.001, help="优化器学习率")
    opt = parser.parse_args()

    train = SSD_Train()
    model = train.load_weights()
    model.summary()
    train_dataloader, val_dataloader = train.get_dataloader()
    train.train(model, train_dataloader, val_dataloader)
