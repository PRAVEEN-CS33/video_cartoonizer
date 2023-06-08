import cv2
import numpy as np

def cartoonize(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter for edge preservation
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Perform adaptive thresholding to detect edges
    edges = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

    # Convert frame to RGB and scale the values
    quantized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    quantized = cv2.convertScaleAbs(quantized, alpha=(255.0/254.0))

    # Convert frame to LAB color space
    quantized = cv2.cvtColor(quantized, cv2.COLOR_RGB2LAB)

    # Apply median blur to smooth the image
    quantized = cv2.medianBlur(quantized, 5)

    # Reshape the image for K-means clustering
    Z = quantized.reshape((-1,3))
    Z = np.float32(Z)

    # Define criteria and perform K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert center values to uint8
    center = np.uint8(center)

    # Apply K-means clustering to quantized image
    res = center[label.flatten()]
    quantized = res.reshape((quantized.shape))

    # Convert quantized image back to RGB
    quantized = cv2.cvtColor(quantized, cv2.COLOR_LAB2RGB)

    # Apply bitwise AND operation with edges to get cartoon effect
    cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)

    return cartoon

cap = cv2.VideoCapture('src\Cute Cat.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if ret:
        # Apply cartoonization to the frame
        cartoon = cartoonize(frame)

        # Convert cartoon image to HSV color space
        hsv = cv2.cvtColor(cartoon, cv2.COLOR_RGB2HSV)

        # Modify the hue channel of HSV image
        hsv[:, :, 0] = (hsv[:, :, 0] + 20) % 180

        # Convert modified HSV image back to RGB
        cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Write the cartoonized frame to the output video
        out.write(cartoon)

        # Check for 'q' key press to stop the loop
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release video capture and video writer
cap.release()
out.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()

# Print completion message
print("completed")