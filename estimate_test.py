import cv2
import numpy as np
import os

# Function to estimate optical flow between two frames using the Lucas-Kanade method
def estimate_optical_flow(prev_frame, current_frame):
    # Convert the frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Detect features in the previous frame
    prev_features = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

    # If no features are found, return an empty array
    if prev_features is None:
        return np.array([])

    # Draw the detected features on the current frame for visualization
    for x, y in prev_features.reshape(-1, 2):
        cv2.circle(current_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Display the current frame
    cv2.imshow('Video', current_frame)
    cv2.waitKey(1)  # Display the frame for 1 ms

    # Calculate optical flow
    current_features, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_features, None, **lk_params)

    # Select good features for the previous frame and the current frame
    good_prev = prev_features[status == 1]
    good_current = current_features[status == 1]

    # Calculate the motion vectors
    motion_vectors = good_current - good_prev

    return motion_vectors

# Function to convert motion vectors to pitch and yaw angles
def motion_vectors_to_angles(motion_vectors, focal_length):
    # Calculate the average motion vector
    average_vector = np.mean(motion_vectors, axis=0)

    # Calculate the pitch and yaw angles
    pitch = np.arctan2(average_vector[1], focal_length)
    yaw = np.arctan2(average_vector[0], focal_length)

    return pitch, yaw




# Directory containing the labeled videos
labeled_dir_path = 'labeled/'

# Directory to save the generated labels
test_dir_path = 'test/'

# Create the test directory if it doesn't exist
os.makedirs(test_dir_path, exist_ok=True)

# Loop over all the '.hevc' files in the labeled directory
for filename in os.listdir(labeled_dir_path):
    if filename.endswith('.hevc'):
        # Full path of the video file
        video_path = os.path.join(labeled_dir_path, filename)

        # Load the video
        cap = cv2.VideoCapture(video_path)

        # Initialize a list to store the labels
        labels = []

        # Loop over the frames of the video
        prev_frame = None
        while True:
            # Read the current frame
            ret, current_frame = cap.read()

            # If the frame was not read successfully, then we have reached the end of the video
            if not ret:
                break

            # If this is the first frame of the video, just store it and continue to the next frame
            if prev_frame is None:
                prev_frame = current_frame
                continue

            # Estimate optical flow between the previous frame and the current frame
            motion_vectors = estimate_optical_flow(prev_frame, current_frame)

            # If no motion vectors were found, skip to the next frame
            if motion_vectors.size == 0:
                prev_frame = current_frame
                continue

            # Convert the motion vectors to pitch and yaw angles
            pitch, yaw = motion_vectors_to_angles(motion_vectors, focal_length=910)

            # Append the pitch and yaw angles to the labels
            labels.append([pitch, yaw])

            # Update the previous frame
            prev_frame = current_frame

        # Save the labels to a text file in the test directory
        # The filename of the text file is the same as the video file, but with '.txt' extension
        labels_filename = os.path.splitext(filename)[0] + '.txt'
        labels_path = os.path.join(test_dir_path, labels_filename)
        np.savetxt(labels_path, labels)

        # Close the video file
        cap.release()