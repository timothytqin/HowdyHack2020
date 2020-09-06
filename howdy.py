import eventlet
import socketio

import os
import time
import csv 
from absl import app, flags, logging
# from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
# flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
#                     'path to weights file')
# flags.DEFINE_integer('size', 416, 'resize images to')
# flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

directory = "./photos"
classes = "./data/coco.names"
weights = "./checkpoints/yolov3.tf"
size = 416
num_classes = 80

width = 1866
height = 1009


sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})

@sio.event
def connect(sid, environ):
    print('connect ', sid)
    parking_spots = []

    with open('parking_lot.csv', mode ='r') as file: 
        csvFile = csv.reader(file) 
        next(csvFile)
        for lines in csvFile: 
            parking_spots.append(Polygon([(int(lines[1]), int(lines[2])), (int(lines[3]), int(lines[4])), (int(lines[5]), int(lines[6])), (int(lines[7]), int(lines[8]))]))

    num_parking_spots = len(parking_spots)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    yolo = YoloV3(classes=num_classes)

    yolo.load_weights(weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(classes).readlines()]
    logging.info('classes loaded')

    dirs = os.listdir(directory)
    dirs.sort(reverse=True)

    for filename in dirs:
        if filename.endswith(".jpg"):
            t1 = time.time()
            img_path = f"{directory}/{filename}"
            img_raw = tf.image.decode_image(
                open(img_path, 'rb').read(), channels=3)

            img = tf.expand_dims(img_raw, 0)
            img = transform_images(img, size)

            boxes, scores, classes, nums = yolo(img)

            empty = [True] * num_parking_spots
            for i in range(num_parking_spots):
                for j in range(nums[0]):
                    # is car
                    if int(classes[0][j]) == 2:
                        arr = np.array(boxes[0][j])
                        ax = arr[0] * width
                        ay = arr[1] * height
                        bx = arr[2] * width
                        by = arr[3] * height
                        mid = Point((ax + bx) / 2, by * 0.98)
                        empty[i] &= not parking_spots[i].contains(mid)
            
            # img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
            # img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
            # cv2.imwrite(f"output/output-{filename}", img)
            
            print(f"{filename}: {empty}")
            t2 = time.time()
            print(t2 - t1)
            sio.emit('frame', empty)


@sio.event
def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)