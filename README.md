# Object Detection Video Processing Pipeline

This Python script processes videos using a YOLO object detection model, creating an output video with bounding boxes around detected objects and a JSON file containing timestamps for each detection.

## Features

- Process videos using a custom YOLO model
- Generate output video with bounding boxes and labels
- Create a JSON file with timestamps for each detected object
- Customizable detection threshold

## Requirements

- Python 3.6+
- OpenCV (cv2)
- Ultralytics YOLO
- A trained YOLO model (.pt file)

## Installation

1. Clone this repository or download the script.
2. Install the required packages:

   pip install opencv-python ultralytics

3. Place your trained YOLO model (.pt file) in a known location.

## Usage

1. Update the following variables in the `main()` function:
   - `VIDEOS_DIR`: Directory containing input videos
   - `MODEL_PATH`: Path to your YOLO model file
   - `VIDEO_NAME`: Name of the input video file

2. Run the script:

   python object_detection_pipeline.py

3. The script will create an output folder named `{video_name}_output` containing:
   - Processed video: `{video_name}_out.mp4`
   - JSON file with detections: `{video_name}_detections.json`

## Output

### Video

The output video will have bounding boxes drawn around detected objects, along with their class labels.

### JSON

The JSON file will contain timestamps for each detected object, organized by class label:

{
    "LABEL1": [10.1, 10.2, 10.3, ...],
    "LABEL2": [20.3, 31.8, 40.12, ...],
    ...
}

## Customization

- Adjust the `threshold` parameter in the `process_frame()` function to change the detection confidence threshold.
- Modify the `process_frame()` function to change the appearance of bounding boxes and labels.

## Script Overview

import os
import json
from ultralytics import YOLO
import cv2

def load_model(model_path):
    return YOLO(model_path)

def process_frame(frame, model, threshold=0.01, frame_count=0, fps=30):
    # Process frame and return detections
    ...

def process_video(input_path, output_folder, model):
    # Process video, save output, and generate JSON
    ...

def main():
    VIDEOS_DIR = r'C:\pepsi_test\test_vids'
    MODEL_PATH = r'C:\Users\rohit\Downloads\comb200.pt'
    VIDEO_NAME = '6.mp4'
    
    model = load_model(MODEL_PATH)
    
    input_path = os.path.join(VIDEOS_DIR, VIDEO_NAME)
    output_folder = os.path.join(VIDEOS_DIR, f'{os.path.splitext(VIDEO_NAME)[0]}_output')
    
    process_video(input_path, output_folder, model)
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

## License

[Specify your chosen license here]

## Contributing

We welcome contributions to improve this object detection pipeline. Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Troubleshooting

- If you encounter "ModuleNotFoundError", make sure you have installed all required packages.
- Ensure that your YOLO model file (.pt) is in the correct location and format.
- Check that the input video path is correct and the video file exists.

## Contact

For questions or support, please contact [Your Name] at [your.email@example.com].

## Acknowledgments

- This project uses the Ultralytics YOLO implementation.
- Thanks to the OpenCV community for their excellent computer vision library.
