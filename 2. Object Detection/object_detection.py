import os
import time
from ultralytics import YOLO

# Start timing
start_time = time.time()

# Set execution path
execution_path = os.getcwd()

# Load the Tiny YOLOv3 model with auto-download feature
model = YOLO('yolov3-tiny.pt')  # This will automatically download the model if not found

# Define input and output video paths
# input_video_path = os.path.join(execution_path, "path_to_your_video.mp4")
input_video_path = r"input\video1.mp4"  # Adjust this path as necessary
output_video_path = os.path.join(execution_path, "output/output_video.mp4")

results = model.predict(source=input_video_path, 
                         save=True, 
                         save_txt=True,  # Set to False to avoid saving text files
                         project=execution_path, 
                         name='output', 
                         conf=0.3,
                         show=True)  # Confidence threshold set to 30%

# End timing and calculate duration
end_time = time.time()
print("Time taken to run the code:", end_time - start_time, "seconds")