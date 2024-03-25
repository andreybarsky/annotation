# annotate a subset of image-question pairs from json file

import os
import sys
import json
import cv2
from annotator import AnnotationSession
import config


json_files = sorted([filename for filename in os.listdir(config.json_dir) if 'json' in filename])

print('Select an annotation set to process:')
for i, filename in enumerate(json_files):
    print(f'{i+1}: {filename}')

try:
    selection = int(input('> '))
    assert selection in range(1, len(json_files)+1)
except:
    print(f'Expected an integer between 1 and {len(json_files)}, please try again')
    sys.exit()

filepath = os.path.join(config.json_dir, json_files[selection-1])
with open(filepath) as chosen_file:
    data = json.load(chosen_file)['data']

image_names = [d['imageNew'] for d in data]

print(data)

if not os.path.exists(config.label_dir):
    print(f'Creating label directory: {config.label_dir}')
    os.mkdir(config.label_dir)

sess = AnnotationSession(image_dir=config.image_dir, label_dir=config.label_dir, image_names=image_names)

sess.help_message()
i = 0
while i < len(sess.image_queue):
    img_path = sess.image_queue[i]
    img_name = img_path.split('/')[-1]

    record = data[i]

    sess.current_image_name = img_name
    print(f'Loading image: {img_name}')
    print(f'  (#{i+1} of {len(sess.image_queue)} in queue)')


    print(f'Question ID: {record["questionId"]}')
    print(f'Question types: {record["question_types"]}')
    print(f'  Question: {record["question"]}')
    print(f'  Answer/s: {"\n    ".join(record["answers"])}')




    signal = sess.process_image(img_path)
    if signal == 'quit':
        break
    elif signal == 'prev':
        # go back along the image list:
        i -= 1
    else:
        # otherwise, go forward
        i += 1



cv2.destroyAllWindows()
