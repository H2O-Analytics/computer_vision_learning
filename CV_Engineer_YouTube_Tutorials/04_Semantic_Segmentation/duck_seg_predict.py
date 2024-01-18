from ultralytics import YOLO
import cv2


model_path = '/Users/tawate/Documents/H2O_Analytics/computer_vision_learning/runs/segment/train/weights/best.pt'

image_path = '/Users/tawate/Documents/H2O_Analytics/data/cv_engineer_youtube_data/semantic_segmentation/duck_segmentation/images/validation/0b515c095d334a56.jpg'
img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

for result in results:
    for j, mask in enumerate(result.masks.data):
        # apply mask to input image
        mask = mask.numpy() * 255
        # make the mask the size of the image
        mask = cv2.resize(mask, (W, H))
        # write output file
        cv2.imwrite('/Users/tawate/Documents/H2O_Analytics/data/cv_engineer_youtube_data/semantic_segmentation/duck_segmentation//output.png', mask)