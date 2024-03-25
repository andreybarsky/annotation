# annotation
Annotation system for labelling bounding boxes using openCV

To install, just requires numpy and cv2:
```pip install numpy opencv-python```

## Instructions

Edit `config.py` to point at the correct directory for json files and images, and specify wherever you want the labels to be saved.

Then just run `annotate_json.py`. It will ask you which set you want to annotate, and open an interactive cv2 window.
You will be presented with an image, and the relevant question-answer pair will be shown in the terminal.
You can draw bounding boxes by clicking and dragging, and delete bounding boxes by right-clicking on them or pressing `d`.
Press `n` to go to the next image, `p` for the previous image, or `q` to finish and exit the program.

cv2 does not behave very well if you close the image window manually or attempt to KeyboardInterrupt out, so using `q` to close the window is recommended.

Annotations are saved as `.npy` arrays in the specified label directory, named after the questionIds defined in the json file.

Please let me know if you encounter any problems.
