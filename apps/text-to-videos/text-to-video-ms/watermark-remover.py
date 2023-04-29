import cv2
import numpy as np

input_video_path = 'concatenated_video.mp4'
output_video_path = 'output_video.mp4'

# Load the video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Define the mask for the watermark region (replace this with the actual watermark position and size)
mask = np.zeros((height, width), np.uint8)
mask[100:200, 100:300] = 255

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Inpaint the watermark region
    inpainted_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

    # Write the frame
    out.write(inpainted_frame)

    cv2.imshow('frame', inpainted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()