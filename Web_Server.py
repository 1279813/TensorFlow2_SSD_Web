# coding:UTF-8
import base64
from io import BytesIO

from PIL import Image
from flask import Flask, request, render_template

from predict import SSD

app = Flask(__name__)

ssd = SSD()
image_list = ["jpg", "png", "jpeg", "pbm", "pgm", "ppm", "tif", "tiff"]
video_list = ["mp4", "mov"]


@app.route('/')
def index():
    return render_template("index.html")


def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str


@app.route('/img', methods=['POST'])
def detect():
    img = request.files['file']
    if img.filename.split(".")[1] in image_list:
        img = Image.open(img)
        frame, txt = ssd.detect_image(img)
        img_stream = im_2_b64(frame)
        return render_template('index.html',
                               img_stream=img_stream.decode("utf-8"),
                               res=txt)
    else:
        return render_template('index.html',
                               img_stream="none",
                               res="请不要上传jpg、png、jpeg以外的格式")


if __name__ == '__main__':
    app.run(host="0.0.0.0",
            port=80,
            debug=True)
