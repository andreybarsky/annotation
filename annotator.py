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

        self.params = xmin, xmax, ymin, ymax

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
        new_bbox = self.__class__(*self.params)
        new_bbox.xmin = self.xmin * factor
        new_bbox.xmax = self.xmax * factor
        new_bbox.ymin = self.ymin * factor
        new_bbox.ymax = self.ymax * factor
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
        if type(class_id) == int:
            self.class_num = class_id
            # self.class_name = self.num2name[class_id]
        elif type(class_id) == str:
            # self.class_name = class_id
            self.class_num = self.name2num[class_id]
        else:
            raise TypeError('Final argument to BoundingBox must be class name or number')

        super().__init__(self, xmin, xmax, ymin, ymax)

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
        return f"<{self.name}: x: {self.xmin:.3}-{self.xmax:.3}; y: {self.ymin:.3}-{self.ymax:.3}>"

    def draw(self, image):
        """takes an image, returns it with this bbox drawn on it"""
        # draw bbox as with generic bbox:
        image = super().draw(self, image)
        # but then staple the class name on top as well:
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

        bounds_list = [bbox.bounds for bbox in self.bboxes]
        arr = np.asarray(bounds_list)
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
                xmin, xmax, ymin, ymax = row
                bbox = GenericBoundingBox(xmin, xmax, ymin, ymax)
            elif len(row) == 5:
                # classful bounding box
                xmin, xmax, ymin, ymax, class_num = row
                bbox = ClassBoundingBox(xmin, xmax, ymin, ymax, class_num)
            bboxes.append(bbox)
        return bboxes


class AnnotationSession(object):
    """interactive user session within which we annotate multiple files"""

    def __init__(self, image_dir, label_dir, max_display_size=config.max_display_size,
                       start_from=0, classes=False, image_names=None, label_names=None):
        """accepts a list of filepaths to images for annotating, and begins a session to annotate them.
        if image_names are given, loop through only those images in the target directory."""
        self.image_dir = image_dir
        self.label_dir = label_dir

        if image_names is None:
            image_names = os.listdir(image_dir)
        # otherwise, assume image_names is a list of filename strings (without leading directories)

        if label_names is None:
            # by default, label names are just the same as image names with the file extension stripped:
            self.image_labels = {imname: '.'.join(imname.split('.')[:-1]) for imname in image_names}
            # but they can be provided manually instead, for e.g. image-question pairs
        else:
            assert len(image_names) == len(label_names)
            self.image_labels = {image_names[i]: label_names[i] for i in range(len(image_names))}

        image_queue = [os.path.join(image_dir, filename) for filename in image_names if filename != 'labels']
        if start_from > 0: # start at a pre-determined index but loop back again
            self.image_queue = image_queue[start_from:] + image_queue[:start_from]
        else:
            self.image_queue = image_queue
        self.current_image_name = None

        self.max_dims = max_display_size

        self.btn_down = False
        self.current_bbox = None

        self.current_mouse_position = (0,0)
        self.use_classes = classes

    def help_message(self):
        # prints user instructions to console

        print('Annotation session started.')
        print("Left click and drag to draw bounding boxes. Right click a box, or press 'd', to delete it.")
        print("Press 'n' for next image, 'p' for previous, and 'q' to quit.")
        print(f'Progress is saved after each image.')

        if self.use_classes:
            print(f'\nUsing classes: {list(config.defined_classes.keys())}')
            print(f'  Middle click on a box to cycle through class labels.')
            if len(config.class_shortcuts) > 0:
                print('  Or use keyboard shortcuts:')
                for k,v in config.class_shortcuts.items():
                    print(f'   {v}: {k}')




    def load_image(self, filepath):
        """loads an image from filepath and downsamples it to fit inside self.max_dims.
        outputs cv2 image object."""

        cv2.namedWindow('Image', 16)

        img = cv2.imread(filepath, 1)
        # cv2's dimensions are height,width in that order, even though in some places we use x,y:
        self.original_dims = list(reversed(img.shape[:2]))
        x_ratio, y_ratio = self.original_dims[0] / self.max_dims[0], self.original_dims[1] / self.max_dims[1]

        if (x_ratio > 1) or (y_ratio > 1): # resize if image dimensions exceed max dimensions
            self.downsampling_factor = max(x_ratio, y_ratio)
            # more cv2 dimension reversal:
            self.new_dims = int(self.original_dims[0] / self.downsampling_factor), int(self.original_dims[1] / self.downsampling_factor)
            # print(f'Original dims: {self.original_dims}')
            # print(f'Downsampling factor: {self.downsampling_factor:.2}')
            # print(f'New dims: {self.new_dims}')
            cv2.resizeWindow('Image', tuple(self.new_dims))
        else:
            self.downsampling_factor = 1 # no downsampling

        # self.current_image = img
        return img

    def get_annotation(self, img):
        # set up data to send to mouse handler
        self.data = {'img': img.copy()}

        # set the callback function for any mouse event

        cv2.imshow("Image", img)
        cv2.setMouseCallback("Image", self.mouse_handler, self.data)
        signal = self.wait_for_boxes()

        anno = copy.deepcopy(self.current_annotation)
        # del self.current_annotation

        return anno, self.data['img'], signal

    def wait_for_boxes(self):
        # main loop that allows the user to draw boxes,
        # also responds to keypresses etc.

        # call empty mouse handler to refresh bounding boxes:
        self.mouse_handler(None, 0,0,None, self.data)

        done = False

        while not done:
            key = chr(cv2.waitKey(0))
            signal = None

            if key == 'n':
                # finish this image and pass to the next
                done = True
            elif key == 'q':
                done = True
                signal = 'quit'
                print(f'Saving current annotation and quitting session.')
                # cv2.destroyAllWindows()
                # sys.exit()
            elif key == 'd':
                # delete the box at the current mouse position
                image = self.data['img'].copy()
                x,y = self.current_mouse_position
                self.delete_box_at(x,y)
                # call empty mouse handler:
                self.mouse_handler(None, x,y,None, self.data)
            elif key == 'p':
                # send signal to load the previous image instead of the next
                done = True
                signal = 'prev'


            elif self.use_classes and key in config.class_shortcuts:
                x,y = self.current_mouse_position
                box_id = self.current_annotation.which_bbox(x,y)
                if box_id is not None:
                    self.current_annotation[box_id].name = config.class_shortcuts[key]
            else:
                print(f'Detected keypress: {key}, but no behaviour defined')
        return signal


    def mouse_handler(self, event, x, y, flags, data):
        image = self.data['img'].copy()

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
                    if not self.use_classes:
                        bbox = GenericBoundingBox(xmin, xmax, ymin, ymax)
                    else:
                        bbox = ClassBoundingBox(xmin, xmax, ymin, ymax, obj_class)
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

            clr = config.default_colour[::-1] # reversed because CV2 uses BGR not RGB

            cv2.circle(image, (xmin, ymin), 2, clr, 5, 16)
            cv2.circle(image, (xmin, ymax), 2, clr, 5, 16)
            cv2.circle(image, (xmax, ymin), 2, clr, 5, 16)
            cv2.circle(image, (xmax, ymax), 2, clr, 5, 16)
            cv2.rectangle(image, data['pt1'], (x, y), clr, 1)

        elif event == cv2.EVENT_LBUTTONDOWN:
            # start a new box
            self.btn_down = True
            data['pt1'] = x,y
            # image = data['img'].copy()
            # cv2.circle(image, data['pt1'], 2, (255, 255, 255), 5, 16)

        elif event == cv2.EVENT_MBUTTONDOWN:
            # change class of the selected box
            if self.use_classes:
                box_idx = self.current_annotation.which_bbox(x, y)
                if box_idx is not None:
                    self.current_annotation[box_idx].cycle_class()
                    self.changes_made = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            # delete the selected box
            self.delete_box_at(x,y)
            self.changes_made = True
            # box_idx = self.current_annotation.which_bbox(x, y)
            # if box_idx is not None:
            #     del self.current_annotation.bboxes[box_idx]
            #     self.changes_made = True

        elif event == cv2.EVENT_MOUSEMOVE:
            # just store the current mouse position so we can link in keyboard commands
            self.current_mouse_position = x,y

        #else: # no change; so don't waste computation time redrawing the image
        #    redraw = False

        # regardless:
        if redraw:
            self.current_annotation.draw(image)
            cv2.imshow("Image", image)
            # self.data['img'] = image

    def delete_box_at(self, x, y):
        """deletes the box located at mouse coordinates x and y"""
        box_idx = self.current_annotation.which_bbox(x, y)
        if box_idx is not None:
            del self.current_annotation.bboxes[box_idx]
            self.changes_made = True


    def process_queue(self):
        self.help_message()
        i = 0
        while i < len(self.image_queue):
            img_path = self.image_queue[i]
            img_name = img_path.split('/')[-1]
            self.current_image_name = img_name

            label_name = self.label_
            print(f'Loading image: {img_name}')
            print(f'  (#{i+1} of {len(self.image_queue)} in queue)')

            signal = self.process_image(img_path, label_name)
            if signal == 'quit':
                break
            elif signal == 'prev':
                # go back along the image list:
                i -= 1
            else:
                # otherwise, go forward
                i += 1


    def process_image(self, img_path, label_path=None):
        """Loads, resizes, and prompts for annotation of a single example"""

        # take everything after the final slash:
        img_name = img_path.split(os.sep)[-1]
        print(f'Img_name: {img_name}')

        # strip file extension from image name: (i.e. take everything before the final period)
        img_name_noext = '.'.join(img_name.split('.')[:-1])

        img = self.load_image(img_path)

        if label_path is None:
            # by default, label path is based on the image name:
            label_name = f'{img_name_noext}.npy'
            print(f'Label name: {label_name}')
            label_path = os.path.join(self.label_dir, label_name)
        else:
            label_name = label_path.split(os.sep)[-1]
            print(f'Label name: {label_name}')

        if os.path.exists(label_path):
            print(f'Loading existing annotation from: {label_path}')
            anno = Annotation(load_from=label_path)
            if self.downsampling_factor > 1:
                anno = anno.resize(1/self.downsampling_factor)
            self.current_annotation = anno

        else:
            self.current_annotation = Annotation()

        self.changes_made = False
        anno, img, signal = self.get_annotation(img)

        if len(anno) > 0 and self.changes_made:
            if self.downsampling_factor > 1:
                # set annotation to the scale of the original, non-resized image
                anno = anno.resize(self.downsampling_factor)
            anno.save(label_path)
        else:
            print('No changes made to this annotation.')

        return signal
    #
    # def process_record(self, image_path, label_path):
    #     """load a record (like a question-answer pair) and its associated image,
    #     save with respect to the record and not the image"""
    #
    #     # take everything after the final slash:
    #     img_name = img_path.split(os.sep)[-1]
    #     print(f'Img_name: {img_name}')
    #
    #     # strip file extension from image name: (i.e. take everything before the final period)
    #     img_name_noext = '.'.join(img_name.split('.')[:-1])
    #
    #     label_name = f'{img_name_noext}.npy'
    #     print(f'Label name: {label_name}')
    #
    #     img = self.load_image(img_path)
    #
    #     # where this method differs from process_image is where the annotation is stored:
    #
    #     if os.path.exists(label_path):
    #         print(f'Loading existing annotation from: {label_path}')
    #         anno = Annotation(load_from=label_path)
    #         if self.downsampling_factor > 1:
    #             anno = anno.resize(1/self.downsampling_factor)
    #         self.current_annotation = anno
    #
    #     else:
    #         self.current_annotation = Annotation()
    #
    #     self.changes_made = False
    #     anno, img, signal = self.get_annotation(img)
    #
    #     if len(anno) > 0 and self.changes_made:
    #         if self.downsampling_factor > 1:
    #             # set annotation to the scale of the original, non-resized image
    #             anno = anno.resize(self.downsampling_factor)
    #         anno.save(label_path)
    #     else:
    #         print('No changes made to this annotation.')
    #
    #     return signal


if __name__ == '__main__':

    if not os.path.exists(config.label_dir):
        print(f'Creating label directory: {config.label_dir}')
        os.mkdir(config.label_dir)

    sess = AnnotationSession(image_dir=config.image_dir, label_dir=config.label_dir)
    sess.process_queue()

    cv2.destroyAllWindows()
