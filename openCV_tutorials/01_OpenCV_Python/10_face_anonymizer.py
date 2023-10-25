# Run instructions:
# 1. activate/create the following environment: -m venv mp_env && source mp_env/bin/activate
# 2. pip install mediapipe

import os
import argparse
import cv2
import mediapipe as mp # used for face detection. This doesnt work with python 3.9.16 need to revert and create environment

# get the location and detection information for image or frame.
# location data = location and confidence of image
def process_img(img, face_detection):

    # get height and width of image
    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            # adjust for size of original image
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # blur faces only and replace original face with blurred face
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))

    return img


args = argparse.ArgumentParser()
print(args)

args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default=None)

args = args.parse_args()


output_dir = '/Users/tawate/Documents/openCV_tutorials/face_anonymizer_output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

################
# Detect Faces #
################
mp_face_detection = mp.solutions.face_detection

# create face detection object. 
#   model selection = 0 --> detect faces that are close to camera (<= 2 meters)
#   model selection = 1 --> detet faces far from camera
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    if args.mode in ["image"]:
        print('image')
        # read image
        img = cv2.imread(args.filePath)

        # process image
        img = process_img(img, face_detection)

        # save image
        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

    elif args.mode in ['video']:
        print('video')
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                       25,
                                       (frame.shape[1], frame.shape[0]))

        while ret:
            # make image have blur where there is a face detected
            frame = process_img(frame, face_detection)
            # outout frame to video 
            output_video.write(frame)
            # read the next frame
            ret, frame = cap.read()

        cap.release()
        output_video.release()

    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(1)

        ret, frame = cap.read()
        while ret:
            frame = process_img(frame, face_detection)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()

        cap.release()