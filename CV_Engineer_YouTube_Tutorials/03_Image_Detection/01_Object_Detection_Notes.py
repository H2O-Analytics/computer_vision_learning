""" object detection basics:
1. input = image
2. output = bounding box, confidence score, object category (class name)

You are only comparing predictions with your ground truth in object detection

Assumptions:
1. many samples
2. classes equally distributed
3. training images are annotated

Common Metrics:
1. training: 
    loss function
2. validation: 
    IoU (intersection over union)
    mAP (mean average precision)

Loss Function(s)
1. related to the learning process
2. different types of loss functions, complex mathematics
3. lower is better
4. should decrease with epochs

IoU (Intersection over Union)
1. measures detection accuracy
2. ranges between 0 and 1
3. High is Better
4. IoU = Area of Overlap of bounding box / Area of union bounding box
5. Needs bounding box ground truth to compare to

mAP (mean average precision)
1. based on the precision-recall curve
2. precision-recall curve is based on the IoU and detection confidence score
3. recall measures how effectively we can find objects
4. precision measures how well we perform once we find an object
5. higher is better
Precision = TP / (TP + FP)
Recall = TP / total ground truth objects
AP@50 = average precision at IOU threshold of .50
"""

""" 
Detectron2 - https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
1. Facebook research object detection and segementation library
2. Contains different models to choose from in the model zoo github url above
3. 
"""

"""
AWS Rekognition
- AWS object detection tool
- Uses credientials to access the tool. 
- Only a few lines of code to create the bounding box dimensions for each label in a frame.
"""