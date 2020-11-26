import json
import cv2
from yolo.backend.utils.box import draw_scaled_boxes
import os
import yolo
from yolo.frontend import create_yolo
import numpy as np
import time

# 1. create yolo instance
yolo_detector = create_yolo("ResNet50", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], 416)

# 2. load pretrained weighted file
# Pretrained weight file is at https://drive.google.com/drive/folders/1Lg3eAPC39G9GwVTCH3XzF73Eok-N-dER

DEFAULT_WEIGHT_FILE = os.path.join(yolo.PROJECT_ROOT, "weights.h5")
yolo_detector.load_weights(DEFAULT_WEIGHT_FILE)

# 3. Load images

import os
import matplotlib.pyplot as plt

DEFAULT_IMAGE_FOLDER = os.path.join(yolo.PROJECT_ROOT, "tests", "dataset", "svhn", "imgs")

img_files = [os.path.join(DEFAULT_IMAGE_FOLDER, "1.png"), os.path.join(DEFAULT_IMAGE_FOLDER, "2.png")]
imgs = []
for fname in img_files:
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(img)
    plt.figure()
    plt.imshow(img)
    plt.show()

# 4. Predict digit region

THRESHOLD = 0.3
for img in imgs:
    boxes, probs = yolo_detector.predict(img, THRESHOLD)

    # 4. save detection result
    image = draw_scaled_boxes(img,
                              boxes,
                              probs,
                              ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

    print("{}-boxes are detected.".format(len(boxes)))
    plt.figure()
    plt.imshow(image)
    plt.show()

# readme:
# clone https://github.com/penny4860/Yolo-digit-detector
# copy/paste detection_example.ipynb to here
# install stuff; must use h5py<3, or else: https://stackoverflow.com/questions/53740577/does-any-one-got-attributeerror-str-object-has-no-attribute-decode-whi
# install CUDA 10.0
# https://www.tensorflow.org/install/source#gpu
# put test images into DEFAULT_IMAGE_FOLDER

'''
print(json.dumps([{'a':1.0, 'b':2.0}, {'c':'hi', 'd':'bye'}]))
HW2_dict = {"bbox":[1,2,3,4], "score":[0.1,0.5], "label":[4,6]}
HW2_list_of_dicts = []
HW2_list_of_dicts.append(HW2_dict) # test image 1
# ...
HW2_list_of_dicts.append(HW2_dict) # ... test image 13068
print(json.dumps(HW2_list_of_dicts))
with open('test.json','w') as f:
    json.dump(HW2_list_of_dicts, f)
'''
def to_json_list(json_items, bbox, score):
    '''
    json_items: []
    bbox: (N,4) np array
    score: (N,10) np array

    label: (N,10) np array
    '''
    #labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    #json_items = []

    #z = zip(bbox, score)
    #for b, s in z:
        # for k in range N
        # b: bbox[k]
        # s: score[k]

        #d = {"bbox":b.tolist(), "score":s.tolist(), "label":labels[np.argmax(s)]}

        #json_items.append(d)
        #with open('0560841.json', 'w') as f:
        #    json.dump(json_items, f)

    # list(map(tuple,boxes)) // [(74, 26, 99, 58), (96, 25, 123, 56)]
    # list(map(np.max, score)) // [0.8480638, 0.8997333]
    # list(map(lambda x: labels[np.argmax(x)],probs)) // [2, 3]

    labels = np.array([0,1,2,3,4,5,6,7,8,9])
    #labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # re-order bbox (x1,y1,x2,y2)-->(y1,x1,y2,x2)
    idx = np.array([1,0,3,2])
    a = list(map(lambda x: tuple(x[idx].tolist()), bbox))
    b = np.array(list(map(lambda x: np.max(x), score))).tolist()
    #b = np.array(list(map(np.max, score))).tolist()
    c = np.array(list(map(lambda x: labels[np.argmax(x)], score))).tolist()
    d = {"bbox": a, "score": b, "label": c}
    json_items.append(d)
    return json_items

def to_json(json_items):
    with open('0560841.json', 'w') as f:
        json.dump(json_items, f)

# test json functions
json_items = []
for img in imgs:
    boxes, probs = yolo_detector.predict(img, THRESHOLD)
    json_items = to_json_list(json_items, boxes, probs)

to_json(json_items)

def sort_function(x):
    x1 = x.split(os.sep)
    last = x1[-1] # '1234.png'
    x2 = last.split('.')
    filenumber = x2[0] # '1234'
    return int(filenumber) # 1234 // sort by this int

# using pretrained model from above

# change dir
DEFAULT_IMAGE_FOLDER = os.path.join(yolo.PROJECT_ROOT, "tests", "dataset", "svhn", "imgs_all", "test")

# get files
fn_list = []
for file in os.listdir(DEFAULT_IMAGE_FOLDER): # DONE: sort to alphabetical
    if file.endswith(".png"):
        fn = os.path.join(DEFAULT_IMAGE_FOLDER, file)
        fn_list.append(fn)
fn_list = sorted(fn_list, key=sort_function)

# read files into list
imgs = [] # overwrite above
for fn in fn_list:
    img = cv2.imread(fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(img)

# predict and save .json
print('predicting...')
t0 = time.perf_counter()
json_items = []
count = 0
for img, fn in zip(imgs,fn_list):
    boxes, probs = yolo_detector.predict(img, THRESHOLD)
    json_items = to_json_list(json_items, boxes, probs)

    count += 1
    print('predicted img:', count, ' : ', fn)
t1 = time.perf_counter()
to_json(json_items)


print('time elapsed (predict and create json list:', t1-t0)
print('-- done')

# DONE: put images into DEFAULT_IMAGE_FOLDER
# install tensorflow-gpu==1.14.0
# TODO: https://docs.python.org/3/library/timeit.html#timeit-examples
# or
#
# import time
# t0 = time.perf_counter()
# ~~~code blob~~~
# t1 = time.perf_counter()
# print(t1-t0)