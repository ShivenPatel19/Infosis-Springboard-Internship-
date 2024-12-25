import os
import time
import sys
from ultralytics import YOLO

# Start timing
start_time = time.time()

# Get input and output paths from arguments
if len(sys.argv) != 3:
    print("Usage: python object_detection.py <input_video_path> <output_video_path>")
    sys.exit(1)

input_video_path = sys.argv[1]
output_video_path = sys.argv[2]

# Ensure output directory exists
output_dir = os.path.dirname(output_video_path)
os.makedirs(output_dir, exist_ok=True)

# Load the Tiny YOLOv3 model
model = YOLO('yolov3-tiny.pt')  # This will auto-download the model if not found

# Run the prediction
results = model.predict(
    source=input_video_path,
    save=True,
    project=output_dir,  # Save output in the specified directory
    name='',  # No additional sub-folder
    conf=0.3,  # Confidence threshold set to 30%
    show=False  # Set to False for backend processing
)

# Rename the generated output file to match the desired output path
# Assuming YOLO generates output file with a fixed naming convention
generated_output = os.path.join(output_dir, "predict.mp4")
if os.path.exists(generated_output):
    os.rename(generated_output, output_video_path)

# End timing and print duration
end_time = time.time()
print("Time taken to run the code:", end_time - start_time, "seconds")
