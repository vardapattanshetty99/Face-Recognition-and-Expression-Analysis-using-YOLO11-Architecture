# This line imports the YOLO class from the ultralytics library, which is necessary to work with YOLO models.
from ultralytics import YOLO

# This line loads a trained model from a specific file path on a local drive (E:/Sem 5/ML project/full_1.pt). This model is ready for inference (making predictions on new data).
model = YOLO('E:/Sem 5/ML project/full_1.pt')

# This line uses the loaded model to perform object detection. The arguments passed to model() configure the inference as follows:
# source=0: Specifies that the input source for the model should be a webcam.
# show=True: Indicates that the results, including detected objects with bounding boxes, should be displayed in a window.
# conf=0.3: Sets the confidence threshold to 0.3, meaning the model will only display detections that it is at least 30% confident about.
# save=True: Instructs the program to save the output video or images with the detections drawn on them.
results = model(source=0, show=True, conf=0.3, save=True)