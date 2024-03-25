

# set the local directories of where images and annotations are to be stored:
json_dir =  '/home/abarsky/data/annotation_sets'
image_dir = '/home/abarsky/data/annotation_sets/imgs'
label_dir = '/home/abarsky/data/annotation_sets/boxes'

# maximum size of displayed images on screen: reduce this if the images do not
# fit on your screen, or increase it if they are too small to read:
max_display_size= (1500,900)

# set the default colour of generic (classless) bounding boxes, in integer RGB:
default_colour = (255, 100, 0) # orange ish





#### parameters below are only relevant for multi-class annotation (untested after refactoring)

# this dict maps class label names to the colours of their bounding boxes.
# to add a new class, you can add a new dict entry:
defined_classes = {'Class A': (0, 255, 255),
                   'Class B': (255, 0, 255)}
# (these 'Class A' and 'Class B' are just examples)

# optionally, you can add keyboard shortcuts for quick selection of specified classes:
class_shortcuts = {'a' : 'Class A',
                   'b' : 'Class B'}
# though note that the hotkeys 'n' and 'q' are reserved (for 'next' and 'quit')
