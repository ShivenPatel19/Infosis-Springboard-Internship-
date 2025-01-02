
# MT Infosis Web Application

This is a Flask-based web application developed as part of the MT Infosis project. It provides multiple functionalities for image and video processing, including tasks like image processing, object detection, object tracking, dense optical flow, and sparse optical flow. The backend is implemented in Flask, and the frontend is powered by HTML templates.

## Features

### 1. Image Processing
- Upload an image, and the application splits it into its red, green, and blue channels.
- Outputs three separate images highlighting the respective color channels.

### 2. Object Detection
- Upload an image or video, and the application detects objects using the YOLO model.
- Generates a video with bounding boxes around detected objects.
- Converts output from AVI to MP4 for better compatibility.

### 3. Object Tracking
- Upload a video, and the application tracks objects across frames using the YOLO model.
- Provides a video output with objects tracked in real time.
- Converts output from AVI to MP4 format.

### 4. Dense Optical Flow
- Upload a video, and the application computes dense optical flow using the Farneback method.
- Generates a visual representation of motion in the video.
- Converts the output to MP4 format.

### 5. Sparse Optical Flow
- Upload a video, and the application computes sparse optical flow using the Lucas-Kanade method.
- Tracks key points and their motion across frames.
- Outputs a video with tracked points and motion paths in MP4 format.

## How to Run the Application

### Prerequisites
1. Python 3.8 or above.
2. Required Python packages (install using `pip`):
   ```bash
   pip install Flask opencv-python-headless numpy ultralytics
   ```
3. FFmpeg installed for video format conversion (use package manager like apt, yum, or brew).

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/mt-infosis-app.git
   cd mt-infosis-app
   ```

2. Create the required directories for uploads and results:
   ```bash
   mkdir uploads results static/results
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://127.0.0.1:5000/`.

### File Structure
```
app.py              # Main Flask application
templates/          # HTML templates for the web application
static/             # Folder for static files (e.g., processed videos, images)
uploads/            # Directory for uploaded files
results/            # Directory for processed outputs
```

### Usage
- Visit the respective routes from the homepage to upload files and view results for each task:
  - `/image_processing` - Process images for RGB channels.
  - `/object_detection` - Detect objects in images or videos.
  - `/object_tracking` - Track objects in videos.
  - `/dense_optical_flow` - Compute dense optical flow for videos.
  - `/sparse_optical_flow` - Compute sparse optical flow for videos.

### Output
- Results are saved in the `static/results/` directory and are accessible for download.

## Notes
- Ensure FFmpeg is correctly installed and accessible via the command line.
- The YOLO model weights (`yolov3-tiny.pt` and `yolov8n.pt`) will be automatically downloaded if not available.

## Acknowledgments
- Built with Flask for backend functionality.
- OpenCV and YOLO models for image and video processing.
- FFmpeg for video format conversion.
