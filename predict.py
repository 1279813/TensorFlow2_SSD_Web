import argparse
import colorsys

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from nets.ssd import SSD300
from utils.anchors import get_anchors
from utils.utils import cvtColor, get_classes, resize_image
from utils.utils_bbox import BBoxUtility


class SSD(object):
    def __init__(self):
        # 获取类别以及类别个数
        self.class_names, self.num_classes = get_classes(opt.classes)
        # 先验框
        self.anchors = get_anchors(opt.input_shape, opt.anchors_size)
        self.num_classes = self.num_classes + 1
        self.bbox_util = BBoxUtility(self.num_classes, nms_thresh=opt.nms_iou)
        # 给每个类别生成一个预测框的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        hsv_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), hsv_tuples))
        # 加载网络
        self.ssd = SSD300([opt.input_shape[0], opt.input_shape[1], 3], self.num_classes)
        self.ssd.load_weights(opt.model, by_name=True)

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (opt.input_shape[1], opt.input_shape[0]), opt.letterbox_image)
        image_data = preprocess_input(np.expand_dims(np.array(image_data, dtype='float32'), 0))
        preds = self.ssd(image_data, training=False).numpy()
        results = self.bbox_util.decode_box(preds,
                                            self.anchors,
                                            image_shape,
                                            opt.input_shape,
                                            opt.letterbox_image,
                                            confidence=opt.confidence)

        if len(results[0]) <= 0:
            txt = "🚀检测结果：没检测到人脸"
            print(txt)
            return image, txt

        classes = np.array(results[0][:, 4], dtype='int32')
        scores = results[0][:, 5]
        obj_boxes = results[0][:, :4]
        font = ImageFont.truetype(font='font/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // opt.input_shape[0], 1)

        for i, c in list(enumerate(classes)):
            print(i, c)
            predicted_class = self.class_names[int(c)]
            box = obj_boxes[i]
            score = round(scores[i], 2)

            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = f"{predicted_class} {score}"
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode("utf-8")

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
            txt = f"🚀检测结果：{str(label, 'UTF-8')}，坐标：{top},{left},{bottom},{right}"
            print(txt)

        return image, txt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="weights/model.h5", help="模型保存位置")
    parser.add_argument('--classes', default="classes/classes.txt", help="预测标签类别")
    parser.add_argument('--img', default="img/test.jpg", help="待预测图片")
    parser.add_argument('--videocap', default=False, help="是否需要调用摄像头检测")
    parser.add_argument('--confidence', default=0.4, help="检测阈值")
    parser.add_argument('--nms_iou', default=0.45, help="非极大抑制所用到的nms_iou大小")
    parser.add_argument('--input_shape', default=[300, 300], help="预测的图像大小，传入列表")
    parser.add_argument('--anchors_size', default=[21, 45, 99, 153, 207, 261, 315], help="指定先验框的大小")
    parser.add_argument('--letterbox_image', default=False, help="是否使用letterbox_image对输入图像进行不失真的resize")
    opt = parser.parse_args()

    ssd = SSD()

    if opt.videocap == True:
        cap = cv2.VideoCapture(0)
        while True:
            open, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            img, txt = ssd.detect_image(frame)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("capture", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    elif opt.videocap == False:
        img = opt.img
        image = Image.open(img)
        img, txt = ssd.detect_image(image)
        img.show()

    else:
        raise Exception("videocap参数异常")
