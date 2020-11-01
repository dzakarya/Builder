
######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a video.
# It draws boxes and scores around the objects of interest in each frame
# of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from tensorflow.keras.models import Sequential, load_model
from keras.applications.resnet50 import preprocess_input
from imutils.video import VideoStream,FileVideoStream
# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
import time

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
VIDEO_NAME = 'abi_eval.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','mscoco_label_map.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = VideoStream("rtsp://admin:Admin123@192.168.0.5:554/Streaming/Channels/101").start()
model=load_model('enetapd.h5')

def predictin(imgin):
    s = cv2.resize(imgin, (300, 300))
    im = np.array(s)
    im = im.astype('float32')
    img = np.expand_dims(im, axis=0)
    output = model.predict(img)
    if output[0][0] > output[0][1]:
        return ("safe"),output[0][0]
    else:
        return ('notsafe'), output[0][1]


font = cv2.FONT_HERSHEY_SIMPLEX
img=0
start_time = time.time()
x = 1 # displays the frame rate every 1 second
counter = 0
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920,1080))
while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = video.read()

    im_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(im_input, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    coordinate = vis_util.return_coordinates(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=5,
        min_score_thresh=0.5,
        skip_scores=True)
    for box in coordinate:
        xmin = box[0]
        xmax = box[1]
        ymin = box[2]
        ymax = box[3]
        st = box[4]
        status = st[0]
        if status == "person":
            feed = image[ymin:ymax, xmin:xmax]
            # cv2.imwrite("safe/"+str(img)+".jpg",feed)
            apdstat,scr = predictin(feed)
            if apdstat == 'safe':
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            else:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            cv2.putText(image, "APD Status = {} {}".format(apdstat,scr), (xmin, ymin), font, 1,
                        (255, 255, 255), 2)

        img=img+1
    #out.write(image)
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', image)
    out.write(image)

    counter += 1
    if (time.time() - start_time) > x:
        print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
out.release()
video.release()
cv2.destroyAllWindows()
