from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
import time
import shutil
import subprocess

app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# RESULT_FOLDER = 'results'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULT_FOLDER'] = RESULT_FOLDER

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# app.config['UPLOAD_FOLDER'] = 'uploads/'
# app.config['RESULT_FOLDER'] = 'results/'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image_processing', methods=['GET', 'POST'])
def image_processing():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image
        img = cv2.imread(filepath)
        if img is None:
            return "Invalid image file", 400
        b, g, r = cv2.split(img)

        # Create green, red, and blue channel images
        green_image = np.zeros_like(img)
        red_image = np.zeros_like(img)
        blue_image = np.zeros_like(img)
        green_image[:, :, 1] = g
        red_image[:, :, 2] = r
        blue_image[:, :, 0] = b

        RESULT_FOLDER = os.path.join('static', 'results')
        app.config['RESULT_FOLDER'] = RESULT_FOLDER

        # Save the images
        cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], 'green.jpg'), green_image)
        cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], 'red.jpg'), red_image)
        cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], 'blue.jpg'), blue_image)

        # Generate RGB plot
        color_distribution = img.mean(axis=(0, 1))
        return render_template('image_processing.html', green='green.jpg', red='red.jpg', blue='blue.jpg')

    return render_template('image_processing.html')

@app.route('/object_detection', methods=['GET', 'POST'])
def object_detection():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Object detection using YOLO
        try:
            from ultralytics import YOLO
            import time

            # Start timing
            start_time = time.time()

            # Load the YOLOv3-tiny model
            model = YOLO('yolov3-tiny.pt')  # Auto-download if not available

            # Define output video path
            output_video_path = os.path.join(app.config['RESULT_FOLDER'], f"detected_{filename}")

            # Run the object detection
            results = model.predict(
                source=filepath,
                save=True,
                save_txt=False,  # Avoid saving text files
                project=app.config['RESULT_FOLDER'],
                name=f"detected_{os.path.splitext(filename)[0]}",
                conf=0.3,  # Confidence threshold
                show=False  # Set to True if you want to visualize the results during processing
            )

            # End timing
            end_time = time.time()
            print("Time taken to process:", end_time - start_time, "seconds")

            # Ensure the processed video path is correct
            output_video_path = os.path.join(
                app.config['RESULT_FOLDER'], 
                f"detected_{os.path.splitext(filename)[0]}/{os.path.splitext(filename)[0]}.avi"
            )

            # After YOLO processing (assuming the result is in .avi format)
            avi_video_path = os.path.join(app.config['RESULT_FOLDER'], f"detected_{os.path.splitext(filename)[0]}/{os.path.splitext(filename)[0]}.avi")
            mp4_video_path = os.path.join(app.config['RESULT_FOLDER'], f"detected_{os.path.splitext(filename)[0]}/{os.path.splitext(filename)[0]}.mp4")

            # Convert the video from .avi to .mp4 using ffmpeg
            subprocess.run(['ffmpeg', '-i', avi_video_path, '-vcodec', 'libx264', '-acodec', 'aac', mp4_video_path])

            # Move the processed MP4 video to the static folder
            static_video_path = os.path.join('static/results', f"object_detection_{os.path.splitext(filename)[0]}.mp4")
            # static_video_path = os.path.join('static/results', f"object_detection_{os.path.splitext(filename)[0]}.mp4")
            os.makedirs(os.path.dirname(static_video_path), exist_ok=True)  # Ensure directory exists
            shutil.move(mp4_video_path, static_video_path)
            return render_template('object_detection.html', video=f"results/object_detection_{os.path.splitext(filename)[0]}.mp4")
        except Exception as e:
            print("Error during object detection:", e)
            return "An error occurred during object detection.", 500

    return render_template('object_detection.html')

@app.route('/object_tracking', methods=['GET', 'POST'])
def object_tracking():
    if request.method == 'POST':
        # Save the uploaded video
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Set up paths for .avi and .mp4 outputs
        avi_output_dir = os.path.join(app.config['RESULT_FOLDER'], f"detected_{os.path.splitext(filename)[0]}")
        os.makedirs(avi_output_dir, exist_ok=True)
        avi_video_path = os.path.join(avi_output_dir, f"{os.path.splitext(filename)[0]}.avi")
        mp4_video_path = os.path.join(avi_output_dir, f"{os.path.splitext(filename)[0]}.mp4")
        static_video_path = os.path.join(app.config['RESULT_FOLDER'], f"object_tracking_{os.path.splitext(filename)[0]}.mp4")

        # Run YOLO object tracking and save as .avi
        try:
            # Load YOLOv8 model
            model = YOLO('yolov8n.pt')

            # Load video
            cap = cv2.VideoCapture(filepath)

            # Get video properties for saving the output
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Set up the video writer to save the output video in .avi format
            output_video = cv2.VideoWriter(
                avi_video_path,
                cv2.VideoWriter_fourcc(*'XVID'),
                fps,
                (frame_width, frame_height)
            )

            # Process each frame
            ret = True
            while ret:
                ret, frame = cap.read()

                if ret:
                    # Detect and track objects
                    results = model.track(frame, persist=True)

                    # Plot the results on the frame
                    frame_ = results[0].plot()

                    # Write the frame to the .avi video
                    output_video.write(frame_)

            # Release resources
            cap.release()
            output_video.release()

            # Convert the .avi video to .mp4 using ffmpeg
            subprocess.run(['ffmpeg', '-i', avi_video_path, '-vcodec', 'libx264', '-acodec', 'aac', mp4_video_path])

            # Move the processed .mp4 video to the static folder
            shutil.move(mp4_video_path, static_video_path)

            # Serve the MP4 video in the response
            return render_template('object_tracking.html', video=f"results/object_tracking_{os.path.splitext(filename)[0]}.mp4")

        except Exception as e:
            return f"Error processing video: {e}", 500

    return render_template('object_tracking.html')
@app.route('/dense_optical_flow', methods=['GET', 'POST'])
def dense_optical_flow():
    if request.method == 'POST':
        # Save the uploaded video
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Set output paths for AVI and MP4 formats
        avi_output_path = os.path.join(app.config['RESULT_FOLDER'], f'dense_flow_{os.path.splitext(filename)[0]}.avi')
        mp4_output_path = os.path.join(app.config['RESULT_FOLDER'], f'dense_flow_{os.path.splitext(filename)[0]}.mp4')

        try:
            # Process the video with dense optical flow
            cap = cv2.VideoCapture(filepath)

            # Check if the video opened successfully
            if not cap.isOpened():
                return "Error: Couldn't open the uploaded video."

            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Initialize VideoWriter for AVI format
            out = cv2.VideoWriter(avi_output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

            # Read the first frame and convert it to grayscale
            ret, first_frame = cap.read()
            if not ret:
                return "Error: Couldn't read the first frame of the video."

            prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the current frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate dense optical flow using Farneback method
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                # Visualize optical flow
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mask = np.zeros_like(frame)
                mask[..., 1] = 255
                mask[..., 0] = angle * 180 / np.pi / 2
                mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

                # Write the frame to the AVI output video
                out.write(rgb)

                # Update the previous frame
                prev_gray = gray

            # Release resources
            cap.release()
            out.release()

            # Convert AVI to MP4 using ffmpeg
            os.system(f"ffmpeg -i {avi_output_path} -vcodec libx264 -acodec aac {mp4_output_path}")

            # Return the MP4 video in the response
            return render_template('dense_optical_flow.html', video=f"results/dense_flow_{os.path.splitext(filename)[0]}.mp4")

        except Exception as e:
            return f"Error processing video: {e}"

    return render_template('dense_optical_flow.html')

@app.route('/sparse_optical_flow', methods=['GET', 'POST'])
def sparse_optical_flow():
    if request.method == 'POST':
        # Save the uploaded video
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Set output paths for AVI and MP4 formats
        avi_output_path = os.path.join(app.config['RESULT_FOLDER'], f'sparse_flow_{os.path.splitext(filename)[0]}.avi')
        mp4_output_path = os.path.join(app.config['RESULT_FOLDER'], f'sparse_flow_{os.path.splitext(filename)[0]}.mp4')

        try:
            # Process the video with sparse optical flow
            cap = cv2.VideoCapture(filepath)

            # Check if the video opened successfully
            if not cap.isOpened():
                return "Error: Couldn't open the uploaded video."

            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Initialize VideoWriter for AVI format
            out = cv2.VideoWriter(avi_output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

            # Parameters for Lucas-Kanade optical flow
            lk_params = dict(winSize=(15, 15), maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            # Parameters for Shi-Tomasi corner detection
            feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

            # Read the first frame and find corners
            ret, old_frame = cap.read()
            if not ret:
                return "Error: Couldn't read the first frame of the video."

            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

            # Create a mask for drawing
            mask = np.zeros_like(old_frame)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate optical flow if there are points to track
                if p0 is not None:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                    if p1 is not None and st is not None:
                        good_new = p1[st == 1]
                        good_old = p0[st == 1]

                        # Draw the tracks
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

                        img = cv2.add(frame, mask)

                        # Update the previous frame and points
                        old_gray = frame_gray.copy()
                        p0 = good_new.reshape(-1, 1, 2)
                    else:
                        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                else:
                    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                    img = frame

                # Write the frame to the AVI output video
                out.write(img)

            # Release resources
            cap.release()
            out.release()

            # Convert AVI to MP4 using ffmpeg
            os.system(f"ffmpeg -i {avi_output_path} -vcodec libx264 -acodec aac {mp4_output_path}")

            # Return the MP4 video in the response
            return render_template('sparse_optical_flow.html', video=f"results/sparse_flow_{os.path.splitext(filename)[0]}.mp4")

        except Exception as e:
            return f"Error processing video: {e}"

    return render_template('sparse_optical_flow.html')

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
