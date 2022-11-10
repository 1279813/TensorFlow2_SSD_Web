# SSD目标检测Web端部署（SSD+Flask）入门难度
#### SSD项目源码来自：https://github.com/bubbliiiing/ssd-tf2 本项目修改跟删减了源项目的部分代码，如有侵权请联系13189187513删除谢谢
**** 注：此项目仅用于交流学习参考等，不用于商业用途 ****

---
## 1. 开启web服务以及参数
### 运行 Web_Server.py 开启web服务
    --host=ip地址，看情况自己设置默认127.0.0.1 
    --port=端口号,看情况自己设置默认80端口

---

## 2. 训练自己的物体检测模型
### 一、数据集制作
    1、可使用labelme进行标注，以voc格式导出，详细用法略

    2、classes文件夹内存放自己数据集对应的标注类别
    
    2、数据集VOCdevkit结构
        - VOC2007
        - Annotations      存放标注xml文件
        - ImageSets/Main   存放训练索引文件
        - JPEGImages       存放图片

### 二、数据集划分
    1、运行voc_annotation.py
        生成训练索引文件以及划分数据集

### 三、模型训练以及预测
    1、运行train.py 
        训练自己的模型

    2、运行predict.py
        调用自己的模型检测物体

---
