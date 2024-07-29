import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

# Load the TensorFlow model
model = tf.saved_model.load('saved_model')

# Define a function to perform object detection
def detect_objects(model, image):
    # Preprocess the image to match the input requirements of the model
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Perform inference
    detections = model(input_tensor)

    # Extract relevant information from the detections
    num_detections = int(detections.pop('num_detections'))
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy()

    # Only keep detections with a score above a certain threshold
    detection_threshold = 0.5
    selected_indices = np.where(detection_scores >= detection_threshold)[0]
    selected_boxes = detection_boxes[selected_indices]
    selected_scores = detection_scores[selected_indices]
    selected_classes = detection_classes[selected_indices]

    return selected_boxes, selected_scores, selected_classes

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, scores, classes):
    draw = ImageDraw.Draw(image)
    for box, score, cls in zip(boxes, scores, classes):
        ymin, xmin, ymax, xmax = box
        width, height = image.size
        left, right, top, bottom = xmin * width, xmax * width, ymin * height, ymax * height
        draw.rectangle(((left, top), (right, bottom)), outline='red', width=3)
        text = f'{cls}: {score:.2f}'
        text_background = [left, top - 9, left + 40, top]  
        draw.rectangle(text_background, fill='red')
        draw.text((left, top - 10), text, fill='white')
    return image

# Load an image (replace with your image loading code)
image_path = 'test02.jpg'
image = Image.open(image_path)
image_resized = image.resize((320, 320))  # Resize to match model's input size

# Convert image to numpy array
image_np = np.array(image_resized)

# Perform object detection
boxes, scores, classes = detect_objects(model, image_np)

image_with_boxes = draw_boxes(image, boxes, scores, classes)
image_with_boxes.show()
