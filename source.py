import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV
    color_ranges = {
        'Red': ((136, 87, 111), (180, 255, 255)),
        'Green': ((25, 52, 72), (102, 255, 255)),
        'Blue': ((94, 80, 2), (120, 255, 255))
    }

    kernel = np.ones((5, 5), np.uint8)

    for color, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, np.uint8)
        upper_bound = np.array(upper, np.uint8)

        # Create mask for the color
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Apply morphological transformations
        mask = cv2.dilate(mask, kernel)

        # Bitwise-AND mask and original frame
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(result, f"{color} Color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

        # Display the result
        cv2.imshow(f"{color} Detection", result)

    # Exit on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

