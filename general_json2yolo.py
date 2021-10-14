import json
import os
import shutil
import numpy as np
from tqdm import tqdm

path = '../superannotate_test/download/yolo' # path to new yolo data
file = '../superannotate_test/download/COCO_LastTest.json'  # coco json to convert
    
# Import json
with open(file) as f:
    data = json.load(f)

# Create folder for new annotations
if os.path.exists(path):
    shutil.rmtree(path)  # delete output folder
os.makedirs(path)  # make new output folder
os.makedirs(path + os.sep + 'labels')  # make new labels folder

# Write image filenames, their shapes and classes
file_id, file_name, width, height = [], [], [], []
for i, x in enumerate(tqdm(data['images'], desc='Files and Shapes')):
    file_id.append(x['id'])
    #file_name.append('IMG_' + x['file_name'].split('IMG_')[-1])
    file_name.append(x['file_name'])
    width.append(x['width'])
    height.append(x['height'])

    # image filenames
    with open(path + '/img_filenames.txt', 'a') as file:
        file.write('%s\n' % file_name[i])

    # image shapes
    with open(path + '/shapes.txt', 'a') as file:
        file.write('%g, %g\n' % (x['width'], x['height']))

# Write classes
# Create a dictionary that associates classes 0,1,2,.. with json class ids
cats = {}
i = 0
for x in tqdm(data['categories'], desc='Names'):
    cats[x['id']] = i
    i+=1
    with open(path + '/classes.txt', 'a') as file:
        file.write('%s\n' % x['name'])

# Write label files
for x in tqdm(data['annotations'], desc='Annotations'):
    i = file_id.index(x['image_id'])  # image index
    image_name = file_name[i]
    extension = image_name.split('.')[-1]
    label_name = image_name.replace(extension, 'txt')
    # The COCO bounding box format is [top left x, top left y, width, height] 
    box = np.array(x['bbox'], dtype=np.float32).ravel()
    box[[0, 2]] /= width[i]  # normalize x by width
    box[[1, 3]] /= height[i]  # normalize y by height
    #box = [box[[0, 2]].mean(), box[[1, 3]].mean(), box[2] - box[0], box[3] - box[1]]  # Ultralytics code
    box = [box[0] + box[2] / 2, box[1] + box[3] / 2, box[2], box [3]] # KJZ's code
    category_id = cats[x['category_id']]
    
    if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
        with open(path + '/labels/' + label_name, 'a') as file:
            file.write('%g %.6f %.6f %.6f %.6f\n' % (category_id, *box))
    else: 
        print(image_name, ' Error in writing bounding box. The box is smaller then 0')

