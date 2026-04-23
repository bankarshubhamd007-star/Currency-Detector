import cv2
import numpy as np
from PIL import Image

def get_perspective_warp(image):
    """
    Attempts to detect a banknote rectangle and warp it to a flat view.
    Returns (warped_image, success_flag)
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    screen_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # If our approximated contour has four points, we can assume we found the note
        if len(approx) == 4:
            screen_cnt = approx
            break
            
    if screen_cnt is None:
        return image, False
    
    # Perspective Warp
    pts = screen_cnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # Ensure standard orientation (landscape)
    if maxHeight > maxWidth:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        
    return warped, True

def check_blur(image):
    """Checks if the image is too blurry using Laplacian variance."""
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

def get_color_score(image, denomination):
    """
    Very basic color range check for Indian notes.
    In a real app, this would use a more robust HSV histogram comparison.
    """
    # Color ranges (simplified HSV)
    # 500: Stone Grey
    # 2000: Magenta
    # 200: Bright Yellow
    # 100: Lavender
    # 50: Fluorescent Blue
    # ...
    return 10 # Placeholder score

def extract_roi(image, feature_name):
    """
    Returns the expected region for specific features on a straightened note.
    Coordinates are normalized (0-100).
    """
    h, w = image.shape[:2]
    rois = {
        "gandhi": (int(0.5*h), int(0.7*h), int(0.6*w), int(0.9*w)),
        "thread": (0, h, int(0.4*w), int(0.5*w)),
        "watermark": (int(0.1*h), int(0.9*h), int(0.05*w), int(0.3*w)),
        "serial": (int(0.7*h), int(0.95*h), int(0.6*w), int(0.95*w))
    }
    y1, y2, x1, x2 = rois.get(feature_name, (0,h,0,w))
    return image[y1:y2, x1:x2]
