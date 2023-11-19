import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=8):

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def slope_lines(image, lines):

    img = image.copy()
    poly_vertices = []
    order = [0, 1, 3, 2]

    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:

            if x1 == x2:
                pass
            else:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1

                if m < 0:
                    left_lines.append((m, c))
                elif m >= 0:
                    right_lines.append((m, c))

    left_line = np.mean(left_lines, axis=0)
    right_line = np.mean(right_lines, axis=0)

    try:
        for slope, intercept in [left_line, right_line]:

            rows, cols = image.shape[:2]
            y1 = int(rows)

            y2 = int(rows*0.7)

            x1 = int((y1-intercept)/slope)
            x2 = int((y2-intercept)/slope)
            poly_vertices.append((x1, y1))
            poly_vertices.append((x2, y2))
            draw_lines(img, np.array([[[x1, y1, x2, y2]]]))

        poly_vertices = [poly_vertices[i] for i in order]
        cv2.fillPoly(img, pts=np.array(
            [poly_vertices], 'int32'), color=(0, 255, 0))
        return cv2.addWeighted(image, 0.7, img, 0.4, 0.)
    except:
        return image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        line_img = slope_lines(line_img, lines)
        return line_img


def weighted_img(img, initial_img, α=0.1, β=1., γ=0.):

    try:
        lines_edges = cv2.addWeighted(initial_img, α, img, β, γ)
    except:
        lines_edges = initial_img
    return lines_edges


def get_vertices(image):
    rows, cols = image.shape[:2]
    bottom_left = [cols*0.15, rows*0.9]
    top_left = [cols*0.35, rows*0.6]
    bottom_right = [cols*0.85, rows*0.9]
    top_right = [cols*0.65, rows*0.6]

    ver = np.array(
        [[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return ver


def lane_finding_pipeline(image):

    gray_img = grayscale(image)
    smoothed_img = gaussian_blur(img=gray_img, kernel_size=5)
    canny_img = canny(img=smoothed_img, low_threshold=180, high_threshold=240)
    masked_img = region_of_interest(
        img=canny_img, vertices=get_vertices(image))
    houghed_lines = hough_lines(
        img=masked_img, rho=1, theta=np.pi/180, threshold=20, min_line_len=20, max_line_gap=180)
    output = weighted_img(
        img=houghed_lines, initial_img=image, α=0.8, β=1., γ=0.)

    return output


cap = cv2.VideoCapture(
    '/home/aditya/projects/Data-Science-Project/Data-Science-Project-main/videos/stock1.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video file")

# Read until video is completed
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        cv2.imshow('Frame', frame)

    # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
