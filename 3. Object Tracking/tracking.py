from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Load video
video_path = 'input/video.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties for saving the output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up the video writer to save the output video
output_video = cv2.VideoWriter(
    'output/processed_video.avi', 
    cv2.VideoWriter_fourcc(*'XVID'), 
    fps, 
    (frame_width, frame_height)
)

ret = True
# Read frames and process
while ret:
    ret, frame = cap.read()

    if ret:
        # Detect and track objects
        results = model.track(frame, persist=True)

        # Plot the results on the frame
        frame_ = results[0].plot()

        # Show the frame with the detection and tracking
        cv2.imshow('frame', frame_)

        # Write the frame to the output video
        output_video.write(frame_)

        # Exit on pressing 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()