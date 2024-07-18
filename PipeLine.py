import os
import json
from ultralytics import YOLO
import cv2

def load_model(model_path):
    return YOLO(model_path)

def process_frame(frame, model, threshold=0.01, frame_count=0, fps=30):
    results = model(frame)
    detections = {}
    
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0]
        score = result.conf[0]
        class_id = result.cls[0]
        
        if score > threshold:
            label = model.names[int(class_id)].upper()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            
            timestamp = frame_count / fps
            if label not in detections:
                detections[label] = []
            detections[label].append(round(timestamp, 2))
    
    return frame, detections

def process_video(input_path, output_folder, model):
    os.makedirs(output_folder, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    output_video_path = os.path.join(output_folder, f'{video_name}_out.mp4')
    output_json_path = os.path.join(output_folder, f'{video_name}_detections.json')
    
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    all_detections = {}
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, frame_detections = process_frame(frame, model, frame_count=frame_count, fps=fps)
        out.write(processed_frame)
        
        for label, timestamps in frame_detections.items():
            if label not in all_detections:
                all_detections[label] = []
            all_detections[label].extend(timestamps)
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Remove duplicate timestamps and sort
    for label in all_detections:
        all_detections[label] = sorted(list(set(all_detections[label])))
    
    with open(output_json_path, 'w') as f:
        json.dump(all_detections, f, indent=4)

def main():
    VIDEOS_DIR = r'Videos'
    MODEL_PATH = r'model.pt'
    VIDEO_NAME = '2.mp4'
    
    model = load_model(MODEL_PATH)
    
    input_path = os.path.join(VIDEOS_DIR, VIDEO_NAME)
    output_folder = os.path.join(VIDEOS_DIR, f'{os.path.splitext(VIDEO_NAME)[0]}_output')
    
    process_video(input_path, output_folder, model)
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()