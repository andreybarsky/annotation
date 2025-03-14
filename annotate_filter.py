# annotate a subset of image-question pairs from json file

import os
import sys
import json
import cv2
from annotator import AnnotationSession
import config


# json_files = sorted([filename for filename in os.listdir(config.json_dir) if 'json' in filename])

# print('Select an annotation set to process:')
# for i, filename in enumerate(json_files):
#     print(f'{i+1}: {filename}')
#
# try:
#     selection = int(input('> '))
#     assert selection in range(1, len(json_files)+1)
# except:
#     print(f'Expected an integer between 1 and {len(json_files)}, please try again')
#     sys.exit()

img_root_dir = config.filter_img_root_dir
target_root_dir = config.filtered_img_target_dir
label_root_dir = config.label_dir

img_details = {}
all_img_paths = []

which_subdirs = None

# build list of images to loop over and their associated details:
img_subdirs = os.listdir(img_root_dir)
for subdir in img_subdirs:
    # filter only those chosen:
    if which_subdirs is None or subdir in which_subdirs:
        subdir_path = os.path.join(img_root_dir, subdir)
        subdir_image_names = os.listdir(subdir_path)
        # images with leading subdir directory prefix:
        subdir_image_identifiers = [os.path.join(subdir, img_name) for img_name in subdir_image_names]
        # full image filepaths:
        subdir_image_paths = [os.path.join(subdir_path, img_name) for img_name in subdir_image_names]
        all_img_paths.extend(subdir_image_paths)

        for i, ident in enumerate(subdir_image_identifiers):
            img_details[ident] = {'original_img_path': subdir_image_paths[i],
                                  'filtered_img_path': os.path.join(target_root_dir, ident),
                                  'label_path': os.path.join(label_root_dir, ident) + '.npy',
                              }


#### copy filtered images that have labels to fix a previous bug:
label_subdirs = os.listdir(label_root_dir)
for subdir in label_subdirs:
    print(f'{subdir=}')
    subdir_path = os.path.join(label_root_dir, subdir)
    subdir_label_names = os.listdir(subdir_path)
    # the image name just lacks the .npy extension:
    subdir_image_names = [fn[:-4] for fn in subdir_label_names]

    # find out where the image exists and where to move it to:
    subdir_image_paths = [os.path.join(img_root_dir, subdir, img_name) for img_name in subdir_image_names]
    subdir_target_paths = [os.path.join(target_root_dir, subdir, img_name) for img_name in subdir_image_names]
    print(f'{subdir_target_paths=}')
    for path in subdir_target_paths:
        
    break

# choose a subdir to process:
print('Select an subdirectory to process:')
print(f'0: Quit')
for i, subdir in enumerate(img_subdirs):
    print(f'{i+1}: {subdir}')

try:
    selection = int(input('> '))
    if selection == 0:
        print(f'Quitting')
        sys.exit()
    else:
        assert selection in range(0, len(img_subdirs)+1)
except:
    print(f'Expected an integer between 1 and {len(img_subdirs)}, please try again')
    sys.exit()



chosen_subdir_path = os.path.join(img_root_dir, img_subdirs[selection-1])
chosen_filter_path = os.path.join(target_root_dir, img_subdirs[selection-1])
chosen_label_path = os.path.join(label_root_dir, img_subdirs[selection-1])

if not os.path.exists(chosen_label_path):
    print(f'First, making directory: {chosen_label_path}')
    os.makedirs(chosen_label_path)

# image_names = [d['imageNew'] for d in data]

# print(data)
#
# if not os.path.exists(config.label_dir):
#     print(f'Creating label directory: {config.label_dir}')
#     os.mkdir(config.label_dir)

sess = AnnotationSession(image_dir=chosen_subdir_path,
                         label_dir=chosen_label_path,
                         image_names=None,
                         classes=True)

sess.help_message()
i = 0
while i < len(sess.image_queue):
    img_path = sess.image_queue[i]
    img_name = img_path.split('/')[-1]

    img_ident = os.path.join(*(img_path.split('/')[-2:]))
    this_img_details = img_details[img_ident]
    # record = img_details[img_ident]

    # label_path = os.path.join(config.label_dir, str(record["questionId"]) + '.npy')
    assert img_path == this_img_details['original_img_path']
    filter_save_path = this_img_details['filtered_img_path']
    label_path = this_img_details['label_path']



    sess.current_image_name = img_name
    print(f'\nLoading image: {img_name}')
    print(f'  (#{i+1} of {len(sess.image_queue)} in queue)')

    # print(f'Question ID: {record["questionId"]}')
    # print(f'Question types: {record["question_types"]}')
    # print(f'  Question: {record["question"]}')
    # answers_joined = "\n    ".join(record["answers"])
    # print(f'  Answer/s: {answers_joined}')

    signal = sess.process_image(img_path, label_path)
    if (sess.changes_made or signal == 'save') and os.path.exists(label_path):
        # save the filtered image too:
        import shutil
        if not os.path.exists(chosen_filter_path):
            print(f'First, making directory: {chosen_filter_path}')
            os.makedirs(chosen_filter_path)
        print(f'Copying file from:\n{img_path}\nto\n{filter_save_path}')
        shutil.copyfile(img_path, filter_save_path)

    if signal == 'quit':
        break
    elif signal == 'prev':
        # go back along the image list:
        i -= 1
    else:
        # otherwise, go forward
        i += 1

cv2.destroyAllWindows()
