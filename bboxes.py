import numpy as np
import cv2
import os
import sys
import copy
import argparse

import config

class GenericBoundingBox:
    """bounding boxes for 2d images with 4-DoF """

    # default colour of generic bounding boxes is set in config file:
    colour = config.default_colour

    def __init__(self, xmin, xmax, ymin, ymax):
        """initialised with bounding box extent values (in pixel units)"""
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    @property
    def params(self):
        return self.xmin, self.xmax, self.ymin, self.ymax

    @property
    def ltrb(self):
        return self.xmin, self.ymin, self.xmax, self.ymax

    @property
    def bounds(self):
        return self.xmin, self.xmax, self.ymin, self.ymax

    def draw(self, image):
        """takes an image, returns it with this bbox drawn on it"""

        # first, convert fractional bbox bounds to pixel coordinates:
        height, width = image.shape[:2]

        xmin, xmax, ymin, ymax = int(self.xmin), int(self.xmax), int(self.ymin), int(self.ymax)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), self.colour[::-1], 2)
        return image

    def resize(self, factor):
        """resizes x and y dimensions of bounds by scalar multiplication"""
        print(f'Resizing with factor={factor}')
        new_bbox = self.__class__(*self.params)
        new_bbox.xmin = int(np.round(self.xmin * factor))
        new_bbox.xmax = int(np.round(self.xmax * factor))
        new_bbox.ymin = int(np.round(self.ymin * factor))
        new_bbox.ymax = int(np.round(self.ymax * factor))
        return new_bbox

    def __repr__(self):
        return f"<Bbox: x: {self.xmin}-{self.xmax}; y: {self.ymin}-{self.ymax}>"

class ClassBoundingBox(GenericBoundingBox):
    # set up mappings from class numbers to class names etc.
    num_classes = len(config.defined_classes)
    num2name = [name for name, colour in config.defined_classes.items()]
    name2num = {name:num for num, name in enumerate(num2name)}
    num2colour = [colour for name, colour in config.defined_classes.items()]

    """bounding box with a class label"""
    def __init__(self, xmin, xmax, ymin, ymax, class_id):
        try:
            class_id = int(class_id)
            self.class_num = class_id
            # self.class_name = self.num2name[class_id]
        except:
            if type(class_id) == str:
                # self.class_name = class_id
                self.class_num = self.name2num[class_id]
            else:
                raise TypeError('Final argument to BoundingBox must be class name or number')

        super().__init__(xmin, xmax, ymin, ymax)

    @property
    def params(self):
        return self.xmin, self.xmax, self.ymin, self.ymax, self.class_num

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
        return f"<{self.name}: x: {self.xmin}-{self.xmax}; y: {self.ymin}-{self.ymax}>"

    def draw(self, image):
        """takes an image, returns it with this bbox drawn on it"""
        # draw bbox as with generic bbox:
        image = super().draw(image)
        # but then staple the class name on top as well:
        cv2.putText(image, self.name, (self.xmin, self.ymin-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.colour[::-1])
            # make bold by drawing it twice:
        cv2.putText(image, self.name, (self.xmin+1, self.ymin-9), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.colour[::-1])
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
        if isinstance(bbox, (GenericBoundingBox, ClassBoundingBox)):
            self.bboxes.append(bbox)
        else:
            raise TypeError('Argument for Annotation.append must be a BoundingBox object')

    def resize(self, factor):
        """resizes x and y dimensions of bbox bounds by scalar multiplication"""
        new_anno = Annotation()
        for idx, bbox in enumerate(self):
            new_anno.append(bbox.resize(factor))
        return new_anno

    def to_array(self, as_fraction=True, img_dims=None):
        """outputs the bounding boxes associated with this annotation
            as a numpy array of shape (num_boxes, 4).
        values are in pixel units if as_fraction is False, or in range 0-1 if True.
            but if True, we require img_dims to be provided (as h,w) for the rescaling."""

        bounds_list = [[int(p) for p in bbox.params] for bbox in self.bboxes]
        print(f'Saving bounding box: {bounds_list}')
        arr = np.asarray(bounds_list, dtype=np.int32)
        # if as_fraction:
        #     assert img_dims is not None, "img_dims must be provided for fractional rescaling"
        #     assert len(img_dims) == 2, "expected img_dims as integer tuple (height, width) in pixels"
        #     h, w = [int(v) for v in img_dims]
        #     xmin, xmax = arr[:,0] / w, arr[:,1] / w
        #     ymin, ymax = arr[:,2] / h, arr[:,3] / h
        #     arr = np.stack([xmin, xmax, ymin, ymax], axis=1)

        return arr


    def which_bbox(self, x, y):
        """given a click location, return the index of the bounding box that the
        click was in, from newest first"""
        ### this is a very inefficient collision detection algorithm
        ### but it should be ok as we don't expect hundreds of bboxes per image
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
            filedir = os.path.join(*(filepath.split('/')[:-1]))
            if not os.path.exists(filedir):
                print(f'Creating directory: {filedir}')
                os.makedirs(filedir)
            print(f'Saving new label: {filepath}')
        # with open(filepath, 'w') as file:
        bbox_arr = self.to_array()
        np.save(filepath, bbox_arr)
            # file.write(kitti_str)

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
        """load bounding boxes from a numpy array and append to this annotation"""
        # with open(filename, 'r') as file:
        arr = np.load(filename, allow_pickle=True)
        bboxes = []
        for row in arr:
            if len(row) == 4:
                # generic / classless bounding box
                xmin, xmax, ymin, ymax = [int(r) for r in row]
                bbox = GenericBoundingBox(xmin, xmax, ymin, ymax)
            elif len(row) == 5:
                # classful bounding box
                xmin, xmax, ymin, ymax, class_num = [int(r) for r in row]
                bbox = ClassBoundingBox(xmin, xmax, ymin, ymax, class_num)
            bboxes.append(bbox)
        return bboxes
