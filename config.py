

# set the local directories of where images and annotations are to be stored:
# json_dir =  '/home/abarsky/data/annotation_sets'

filter_img_root_dir = '/home/abarsky/data/IVAM/cameras/navi_bordeaux'
filtered_img_target_dir = '/home/abarsky/data/IVAM/real_filtered/images/navi_bordeaux/'
label_dir = '/home/abarsky/data/IVAM/real_filtered/labels/navi_bordeaux/'

# maximum size of displayed images on screen: reduce this if the images do not
# fit on your screen, or increase it if they are too small to read:
max_display_size= (1500,900)

# set the default colour of generic (classless) bounding boxes, in integer RGB:
default_colour = (255, 255, 255) # white





#### parameters below are only relevant for multi-class annotation (untested after refactoring)

# this dict maps class label names to the colours of their bounding boxes.
# to add a new class, you can add a new dict entry:
defined_classes = {'Container': (255, 100, 0),
                   'Damage': (255, 0, 0)}
# (these 'Class A' and 'Class B' are just examples)

# optionally, you can add keyboard shortcuts for quick selection of specified classes:
class_shortcuts = {'c' : 'Container',
                   'x' : 'Damage'}

# the following hotkeys are reserved:
reserved_hotkeys = ['n', # next
                    'p', # prev
                    'd', # delete
                    's', # save (with annotation but no boxes)
                    'q', # quit
]
