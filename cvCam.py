import cv2

# http://sh:sh@192.168.1.3:8080
# rtsp://sh:sh@192.168.1.3:8080
# Initialize the IP camera (adjust the URL and credentials as needed)
camera_url = "rtsp://sh:sh@192.168.1.3:8080/h264_ulaw.sdp"
cap = cv2.VideoCapture(camera_url)

# Initialize variables
motion_detected = False

# print("Высота:" + str(img.shape[0]))
# print("Ширина:" + str(img.shape[1]))
# print("Количество каналов:" + str(img.shape[2]))

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# out = cv2.VideoWriter(
#     "output.avi",
#     cv2.VideoWriter_fourcc("M", "J", "P", "G"),
#     10,
#     (frame_width, frame_height),
# )

while True:
    # Capture frames from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve motion detection
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if not motion_detected:
        # Store the first frame as the baseline for comparison
        baseline_frame = gray.copy()
        motion_detected = True
        continue

    # Compute the absolute difference between the current frame and the baseline frame
    frame_delta = cv2.absdiff(baseline_frame, gray)

    # Apply a threshold to the frame delta to highlight regions with significant changes
    thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Loop over the contours and check for motion
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Adjust this threshold as needed
            motion_detected = True

            # Get the bounding box coordinates of the motion region
            (x, y, w, h) = cv2.boundingRect(contour)

            # Draw a bounding box around the motion region
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Take a screenshot of the moment when motion is detected
            screenshot = frame[y : y + h, x : x + w]
            cv2.imwrite("screenshot.jpg", screenshot)

    cv2.imshow("Motion Detection", frame)
    # out.write(frame)

    # If 'q' is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera, close all windows, and release the video writer
cap.release()
# out.release()
cv2.destroyAllWindows()
