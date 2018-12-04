#!/usr/bin/python3.6

import numpy as np
import cv2
import os
import sys
import copy
import argparse

class BoundingBox(object):
    """bounding boxes for 2d images with /4-DoF and a class label"""

    # to introduce new classes: add them here with a colour to identify them
    names_colours =[('PalletBody', (255, 255, 255)),
                    ('PalletFace', (0,   255, 0  )),
                    ('Pedestrian', (0,   0,   255)),
                    ('Bay',        (255, 0,   255)),
                    ('Load',       (255, 0,   0  )),
                    ('Truck',      (255, 255, 0  )),
                    ('Racking',    (0,   255, 255))]

    # set up mappings from class numbers to class names and vice versa:
    num2name = [name for name, colour in names_colours]
    name2num = {name:num for num, name in enumerate(num2name)}
    num_classes = len(num2name)

    num2colour = [colour for name, colour in names_colours]

    def __init__(self, class_id, xmin, xmax, ymin, ymax):
        """first argument can be the name or number of the desired class"""
        if type(class_id) == int:
            self.class_num = class_id
            # self.class_name = self.num2name[class_id]
        elif type(class_id) == str:
            # self.class_name = class_id
            self.class_num = self.name2num[class_id]
        else:
            raise TypeError('First argument to BoundingBox must be class name or number')

        #self.colour = self.num2colour[self.class_num]

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    @property
    def bounds(self):
        return self.xmin, self.xmax, self.ymin, self.ymax

    @property
    def colour(self):
        return self.num2colour[self.class_num]

    def cycle_class(self):
        """allows us to cycle from 0 to n then back to 0 again"""
        self.class_num = (self.class_num+1) % self.num_classes
        # self.colour = self.num2colour[self.class_num]

    @property
    def name(self):
        return str(self)

    @name.setter
    def name(self, new_name):
        self.class_num = self.name2num[new_name]

    def __str__(self):
        return self.num2name[self.class_num]

    def __repr__(self):
        return f"{self.name}: x: {self.xmin}-{self.xmax}; y: {self.ymin}-{self.ymax}"

    def resize(self, factor):
        """resizes x and y dimensions of bounds by scalar multiplication"""
        new_bbox = BoundingBox(self.class_num, 0, 0, 0, 0)
        new_bbox.xmin = self.xmin * factor
        new_bbox.xmax = self.xmax * factor
        new_bbox.ymin = self.ymin * factor
        new_bbox.ymax = self.ymax * factor
        return new_bbox

    def draw(self, image):
        """takes an image, returns it with this bbox drawn on it"""
        xmin, xmax, ymin, ymax = int(self.xmin), int(self.xmax), int(self.ymin), int(self.ymax)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), self.colour, 2)
        cv2.putText(image, self.name, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.colour)
        return image


class Annotation(object):
    """a container for several bboxes arranged on an image, in order of insertion"""
    def __init__(self, bboxes=None, load_from=None):
        if load_from is not None:
            bboxes = self.load_bboxes_from_file(load_from)
        elif bboxes is None:
            bboxes = []
        self.bboxes = bboxes
    def __len__(self):
        return len(self.bboxes)
    def __getitem__(self, i):
        return self.bboxes[i]
    def __iter__(self):
        return iter(self.bboxes)
    def __reversed__(self):
        return reversed(self.bboxes)

    def append(self, bbox):
        if isinstance(bbox, BoundingBox):
            self.bboxes.append(bbox)
        else:
            raise TypeError('Argument for Annotation.append must be a BoundingBox object')

    def resize(self, factor):
        """resizes x and y dimensions of bbox bounds by scalar multiplication"""
        new_anno = Annotation()
        for idx, bbox in enumerate(self):
            new_anno.append(bbox.resize(factor))
        return new_anno

    def to_kitti(self):
        """takes a list of bboxes and outputs string in KITTI format"""
        """KITTI label file documentation:
        #Values    Name      Description
        ----------------------------------------------------------------------------
           1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
           1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                             truncated refers to the object leaving image boundaries
           1    occluded     Integer (0,1,2,3) indicating occlusion state:
                             0 = fully visible, 1 = partly occluded
                             2 = largely occluded, 3 = unknown
           1    alpha        Observation angle of object, ranging [-pi..pi]
           4    bbox         2D bounding box of object in the image (0-based index):
                             contains left, top, right, bottom pixel coordinates
           3    dimensions   3D object dimensions: height, width, length (in meters)
           3    location     3D object location x,y,z in camera coordinates (in meters)
           1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
           1    score        Only for results: Float, indicating confidence in
                             detection, needed for p/r curves, higher is better.
        """
        rows = []
        for bbox in self:
            l, r, t, b = bbox.bounds
            row_vals = [0, 0, 0, l, t, r, b, 0, 0, 0, 0] # we don't care about the other values for now
            row_vals = [str(v) for v in row_vals]
            row = [bbox.name] + row_vals
            row_str = ' '.join(row)
            rows.append(row_str)
        kitti_str = '\n'.join(rows)
        return kitti_str

    def which_bbox(self, x, y):
        """given a click location, return the index of the bounding box that the
        click was in, from newest first"""
        for idx, bbox in enumerate(reversed(self)): # reversed so we process oldest boxes last
            xmin, xmax, ymin, ymax = bbox.bounds
            if xmin < x < xmax:
                if ymin < y < ymax:
                    return len(self) - idx - 1 # recover the index of the non-reversed list
        return None # no box here

    def save(self, filepath):
        if os.path.isfile(filepath):
            print(f'Overwriting label at: {filepath}')
        else:
            print(f'Saving new label: {filepath}')
        with open(filepath, 'w') as file:
            kitti_str = self.to_kitti()
            file.write(kitti_str)

    def draw(self, image):
        """takes an image, returns it with every bbox drawn on it"""
        for bbox in self:
            image = bbox.draw(image)
        return image

    def __repr__(self):
        replines = ['-----', 'Annotation:']
        for bbox in self:
            replines.append(bbox.__repr__())
        replines.append('-----')
        return '\n'.join(replines)

    def load_bboxes_from_file(self, filename):
        with open(filename, 'r') as file:
            text = file.read()
        lines = text.splitlines()

        bboxes = []
        for line in lines:
            vals = line.split(' ')
            class_name = vals[0]
            rest_vals = [int(float(val)) for val in vals[1:]]
            xmin, ymin, xmax, ymax = rest_vals[3:7]
            bbox = BoundingBox(class_name, xmin, xmax, ymin, ymax)
            bboxes.append(bbox)
        return bboxes

class AnnotationSession(object):
    class_shortcuts = {'b': 'PalletBody',
       'f': 'PalletFace',
       'p': 'Pedestrian',
       's': 'Stillage',
       'r': 'Racking',
       't': 'Truck',
       'l': 'Load'}


    def __init__(self, image_dir, label_dir, max_dims=[800,600], start_from=0):
        """accepts a list of filepaths to images for annotating, and begins a session to annotate them"""
        self.image_dir = image_dir
        self.label_dir = label_dir

        image_queue = sorted([os.path.join(image_dir, filename) for filename in os.listdir(image_dir)])
        if start_from > 0: # start at a pre-determined index but loop back again
            self.image_queue = image_queue[start_from:] + image_queue[:start_from]
        else:
            self.image_queue = image_queue

        self.max_dims = max_dims

        self.btn_down = False
        self.current_bbox = None

        self.current_mouse_position = (0,0)

    def help_message(self):
        """Print user instructions to console"""
        print("""  Annotation session started.
  Press 'n' for next image, and 'q' to quit.

  Class shortcuts:""")
        for k,v in self.class_shortcuts.items():
            print(f'   {v}: {k}')

    def process_image(self, img_path):
        """Loads, resizes, and prompts for annotation of a single example"""
        img_name = img_path.split(os.sep)[-1]
        print(f'Img_name: {img_name}')
        img_name_noext = img_name.split('.')[0]
        label_name = f'{img_name_noext}.txt'
        print(f'Label name: {label_name}')

        img = self.load_image(img_path)

        label_path = os.path.join(self.label_dir, label_name)
        if os.path.exists(label_path):
            print('Loading existing annotation from: %s' % label_path)
            anno = Annotation(load_from=label_path)
            if self.downsampling_factor > 1:
                anno = anno.resize(1/self.downsampling_factor)
            self.current_annotation = anno

        else:
            self.current_annotation = Annotation()

        self.changes_made = False
        anno, img = self.get_annotation(img)

        if len(anno) > 0 and self.changes_made:
            if self.downsampling_factor > 1:
                # set annotation to the scale of the original, non-resized image
                anno = anno.resize(self.downsampling_factor)
            anno.save(label_path)
        else:
            print('No changes made to this annotation.')

    def load_image(self, filepath):
        """loads an image from filepath and downsamples it to fit inside self.max_dims"""
        img = cv2.imread(filepath, 1)
        # note that cv2 reads x and y dimensions in the opposite order from what is usual:
        self.original_dims = list(reversed(img.shape[:2]))
        x_ratio, y_ratio = self.original_dims[0] / self.max_dims[0], self.original_dims[1] / self.max_dims[1]

        if (x_ratio > 1) or (y_ratio > 1): # resize if image dimensions exceed max dimensions
            self.downsampling_factor = max(x_ratio, y_ratio)
            # more cv2 dimension reversal:
            self.new_dims = int(self.original_dims[0] / self.downsampling_factor), int(self.original_dims[1] / self.downsampling_factor)
            # self.new_dims = [int(d / self.downsampling_factor) for d in self.original_dims]
            print(f'Original dims: {self.original_dims}')
            print(f'Downsampling factor: {self.downsampling_factor}')
            print(f'New dims: {self.new_dims}')
            img = cv2.resize(img, tuple(self.new_dims))
        else:
            self.downsampling_factor = 1 # no downsampling

        return img

    def get_annotation(self, img):
        # Set up data to send to mouse handler
        self.data = {'img': img.copy()}

        # Set the callback function for any mouse eventr
        cv2.imshow("Image", img)
        cv2.setMouseCallback("Image", self.mouse_handler, self.data)
        #cv2.waitKey(0)
        self.wait_for_key()

        anno = copy.deepcopy(self.current_annotation)
        # del self.current_annotation

        return anno, self.data['img']

    def wait_for_key(self):
        done = False

        while not done:
            key = chr(cv2.waitKey(0))

            if key == 'n':
                done = True
            elif key == 'q':
                sys.exit()
            elif key in self.class_shortcuts:
                x,y = self.current_mouse_position
                box_id = self.current_annotation.which_bbox(x,y)
                if box_id is not None:
                    self.current_annotation[box_id].name = self.class_shortcuts[key]


    def mouse_handler(self, event, x, y, flags, data):
        image = data['img'].copy()

        redraw = True
        if event == cv2.EVENT_LBUTTONUP and self.btn_down:
            # if we release the button, finish the box
            self.btn_down = False
            # data['pt2'] = x,y

            # format bounding box as xmin, xmax, ymin, ymax (lrtb):
            x1, y1 = data['pt1']
            x2, y2 = x, y # data['pt2']
            xmin = min(x1, x2)
            xmax = max(x1, x2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)
            obj_class = 0
            if xmax-xmin > 5: # protect against stray clicks by enforcing minimum box size
                if ymax-ymin > 5:
                    bbox = BoundingBox(obj_class, xmin, xmax, ymin, ymax)
                    # and save:
                    self.current_annotation.append(bbox)
                    print(self.current_annotation)
                    self.changes_made = True

        elif event == cv2.EVENT_MOUSEMOVE and self.btn_down:
            # visualise the box-in-progress as we draw it
            x1, y1 = data['pt1']
            x2, y2 = x, y
            xmin = min(x1, x2)
            xmax = max(x1, x2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)

            cv2.circle(image, (xmin, ymin), 2, (255,)*3, 5, 16)
            cv2.circle(image, (xmin, ymax), 2, (255,)*3, 5, 16)
            cv2.circle(image, (xmax, ymin), 2, (255,)*3, 5, 16)
            cv2.circle(image, (xmax, ymax), 2, (255,)*3, 5, 16)
            cv2.rectangle(image, data['pt1'], (x, y), (255,)*3, 1)

        elif event == cv2.EVENT_LBUTTONDOWN:
            # start a new box
            self.btn_down = True
            data['pt1'] = x,y
            # image = data['img'].copy()
            # cv2.circle(image, data['pt1'], 2, (255, 255, 255), 5, 16)

        elif event == cv2.EVENT_MBUTTONDOWN:
            # change class of the selected box
            box_idx = self.current_annotation.which_bbox(x, y)
            if box_idx is not None:
                self.current_annotation[box_idx].cycle_class()
                self.changes_made = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            # delete the selected box
            box_idx = self.current_annotation.which_bbox(x, y)
            if box_idx is not None:
                del self.current_annotation.bboxes[box_idx]
                self.changes_made = True

        elif event == cv2.EVENT_MOUSEMOVE:
            # just store the current mouse position so we can link in keyboard commands
            self.current_mouse_position = x,y

        #else: # no change; so don't waste computation time redrawing the image
        #    redraw = False

        # regardless:
        if redraw:
            self.current_annotation.draw(image)
            cv2.imshow("Image", image)

    def process_queue(self):
        self.help_message()
        for img_path in self.image_queue:
            self.process_image(img_path)


parser = argparse.ArgumentParser()
parser.add_argument('-f', help='File number to start from', type=int, default=0)
args = parser.parse_args()

sess = AnnotationSession(image_dir='../datagen/data/images/', label_dir='../datagen/data/labels/', start_from=args.f)
sess.process_queue()
cv2.destroyAllWindows()
