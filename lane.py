import keras.models as km
import cv2
import numpy as np
import torch
# from testfinal import make_box

model = km.load_model(
    '/home/aditya/projects/Data-Science-Project/Data-Science-Project-main/Traffic Signal Rec/model.h5')


class lan():
    def __init__(self):
        self.rec = []
        self.avg = []


lanes = lan()


def roadlines(image):
    s_image = cv2.resize(image, (160, 80))
    s_image = np.array(s_image)
    s_image = s_image[None, :, :, :]

    pred = model.predict(s_image)[0] * 255

    lanes.rec.append(pred)
    if len(lanes.rec) > 5:
        lanes.rec = lanes.rec[1:]

    lanes.avg = np.mean(np.array([i for i in lanes.rec]), axis=0)

    blank = np.zeros_like(lanes.avg).astype(np.uint8)
    lane = np.dstack((blank, blank, lanes.avg))

    (H, W) = image.shape[:2]

    lane_im = cv2.resize(lane, (W, H))
    result = cv2.addWeighted(image, 0.7, lane_im, 0.3,
                             0, dtype=cv2.CV_32F).astype(np.uint8)

    return result


# def run(inp):
#     model = torch.hub.load('ultralytics/yolov5', 'custom',
#                            path="/home/aditya/projects/Data-Science-Project/Data-Science-Project-main/best.pt", force_reload=True)
#     fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#     out = cv2.VideoWriter('output.mp4', fourcc, 50, (640, 360))
#     cap = cv2.VideoCapture(inp)
#     try:

#         while (cap.isOpened()):
#             _, frame = cap.read()
#             results = model(frame)
#             out.write(roadlines(np.squeeze(results.render())))
#             cv2.imshow("result", roadlines(
#                 np.squeeze(results.render())))
#             if cv2.waitKey(60) & 0xFF == ord('q'):
#                 break

#     except:
#         cap.release()
#         out.release()
#         cv2.destroyAllWindows()
#         return
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
# # run('/home/aditya/projects/Data-Science-Project/Data-Science-Project-main/videos/videoplayback.mp4')
