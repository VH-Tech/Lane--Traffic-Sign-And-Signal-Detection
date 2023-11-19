import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import pickle

from lane import roadlines


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

# ----------------------------------------------------------------------------------------------


def getBox(frame, cascade, upperIgnore=0.1, bilateralParams=(9, 75, 75), scale=5, minNeighbors=4):
    c = 1
    (H, W) = frame.shape[:2]
    upperIgnore = int(upperIgnore*H)
    test = frame[upperIgnore:, :]
    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    (a, b, c) = bilateralParams
    test = cv2.bilateralFilter(test, a, b, c)
    faces, rejects, weights = cascade.detectMultiScale3(
        test, scale, minNeighbors, outputRejectLevels=True, minSize=(30, 30), maxSize=(300, 300))
    return faces, weights


def getCascade(filepath):
    return cv2.CascadeClassifier(filepath)


def getVehicleCNN(filepath):
    pickle_in = open(filepath, 'rb')
    model = pickle.load(pickle_in)
    return model


def make_boxes(frame, cascade, model):
    (H, W) = frame.shape[:2]
    upperIgnore = int(0.1*H)
    faces, weights = getBox(frame, cascade, 0.1, scale=4)
    l = []
    c = 1
    notNullBoxes = []
    for (x, y, w, h) in faces:
        try:
            cropped = frame[upperIgnore+y:upperIgnore+y+h, x:x+w]
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            cropped = cv2.resize(
                cropped, (32, 32), interpolation=cv2.INTER_AREA)
            l.append(cropped)
            notNullBoxes.append((x, y, w, h))
        except:
            pass
        c += 1
    l = np.array(l)
    l = l/255
    l = l.reshape((-1, 32, 32, 1))
    if len(l) != 0:
        pred = np.argmax(model.predict(l), axis=1)
        # print(pred)

    counter = 0
    boxes = []
    for (x, y, w, h) in notNullBoxes:
        if pred[counter] == 1:
            frame = cv2.rectangle(frame, (x, upperIgnore+y),
                                  (x+w,  upperIgnore+y+h), (0, 255, 0), 3)
        counter += 1
    return frame


# ----------------------------------------------------------------------------------------------------------------
# pre trained lanes


def run(inp):

    saveFrames = []

    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path="best.pt", force_reload=True)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    cascade = getCascade(
        '/home/aditya/projects/Data-Science-Project/Data-Science-Project-main/Traffic Signal Rec/cascade.xml')
    CNN = getVehicleCNN(
        '/home/aditya/projects/Data-Science-Project/Data-Science-Project-main/Traffic Signal Rec/modelcar.p')

    cap = cv2.VideoCapture(inp)
    ok, frame = cap.read()
    (H, W) = frame.shape[:2]

    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (W, H))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    c = 0

    while True:
        ok, frame = cap.read()
        c += 1
        print(f'{c}/{length}')
        if not ok or c > 1000:
            break
        results = model(frame)
        #stage = lane_finding_pipeline(np.squeeze(results.render()))
        lanescnn = roadlines(np.squeeze(results.render()))
        final = make_boxes(lanescnn, cascade, CNN)

        saveFrames.append(final)

        out.write(final)

    cap.release()
    out.release()
    print('Processing done.')

    for f in saveFrames:
        cv2.imshow("result", f)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# run('/home/aditya/projects/Data-Science-Project/Data-Science-Project-main/videos/NYCcut.mp4')
