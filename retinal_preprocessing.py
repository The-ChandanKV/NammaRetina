"""Image preparation shared by offline training data and online inference."""

import cv2
import numpy as np


def prepare_retina(image, size=224):
    """Crop black borders, resize and apply Ben Graham local-mean enhancement."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if width > 50 and height > 50:
            image = image[y:y + height, x:x + width]
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
    return np.clip(image, 0, 255).astype(np.uint8)
