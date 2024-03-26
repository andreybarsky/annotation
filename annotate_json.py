# annotate a subset of image-question pairs from json file

import os
import sys
import json
import cv2
from annotator import AnnotationSession
import config

# get the list of json files that indicate annotation sets:
json_files = sorted([filename for filename in os.listdir(config.json_dir) if 'json' in filename])

# allow you to choose one:
print('Select an annotation set to process:')
for i, filename in enumerate(json_files):
    print(f'{i+1}: {filename}')

try:
    selection = int(input('> '))
    assert selection in range(1, len(json_files)+1)
except:
    print(f'Expected an integer between 1 and {len(json_files)}, please try again')
    sys.exit()


# load the chosen file:
filepath = os.path.join(config.json_dir, json_files[selection-1])
with open(filepath) as chosen_file:
    data = json.load(chosen_file)['data']
# and get the list of images it describes:
image_names = [d['imageNew'] for d in data]

if not os.path.exists(config.label_dir):
    print(f'Creating label directory: {config.label_dir}')
    os.mkdir(config.label_dir)

# begin annotation session:
sess = AnnotationSession(image_dir=config.image_dir, label_dir=config.label_dir, image_names=image_names)
sess.help_message()

# loop through the image queue:
i = 0
while i < len(sess.image_queue):
    # get the filename of the image:
    img_path = sess.image_queue[i]
    img_name = img_path.split('/')[-1]
    sess.current_image_name = img_name

    print(f'\nLoading image: {img_name}')
    print(f'  (#{i+1} of {len(sess.image_queue)} in queue)')

    # and the other data associated with this record:
    record = data[i]

    # save the annotation as .npy file according to the name of the record's questionID
    label_path = os.path.join(config.label_dir, str(record["questionId"]) + '.npy')


    print(f'Question ID: {record["questionId"]}')
    print(f'Question types: {record["question_types"]}')
    print(f'  Question: {record["question"]}')
    print(f'  Answer/s:\n' + "\n    ".join(record["answers"]))

    signal = sess.process_image(img_path, label_path)
    print(f'Annotation saved at: {label_path}')

    if signal == 'quit':
        break
    elif signal == 'prev':
        # go back along the image list:
        i -= 1
    else:
        # otherwise, go forward
        i += 1

cv2.destroyAllWindows()
