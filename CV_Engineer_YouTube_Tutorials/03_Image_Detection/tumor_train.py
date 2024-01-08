import argparse
import detectron2_util
import config

from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file
import os

cfg = get_cfg()
cfg.merge_from_file(get_config_file("C/Users/tawate/Documents/H2O_Analytics/computer_vision_learning/CV_Engineer_YouTube_Tutorials/03_Image_Detection/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (config.DATA_PATH + "/object_detection/brain_tumors/train/imgs")
cfg.DATASETS.TEST = ("my_dataset_val",)

cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001

cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 3000 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13 #your number of classes + 1

cfg.TEST.EVAL_PERIOD = 500

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# image need to be pre annotated and in the following data structure for this to work
if __name__ == "__main__":
    """
    annotations should be provided in yolo format, this is: 
            class, xc, yc, w, h
    data needs to follow this structure:
    
    data-dir
    ----- train
    --------- imgs (image files)
    ------------ filename0001.jpg
    ------------ filename0002.jpg
    ------------ ....
    --------- anns (annotation files)
    ------------ filename0001.txt
    ------------ filename0002.txt
    ------------ ....
    ----- val
    --------- imgs (image files)
    ------------ filename0001.jpg
    ------------ filename0002.jpg
    ------------ ....
    --------- anns (annotation files)
    ------------ filename0001.txt
    ------------ filename0002.txt
    ------------ ....
    
    """
    # Specifies the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--class-list', default=config.PROGRAM_PATH+'/detectron2_class.names')
    parser.add_argument('--data-dir', default=config.DATA_PATH + '/object_detection/brain_tumors')
    parser.add_argument('--output-dir', default=config.OUTPUT_PATH)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--learning-rate', default=0.00025)
    parser.add_argument('--batch-size', default=4)
    parser.add_argument('--iterations', default=10000)
    parser.add_argument('--checkpoint-period', default=500)
    parser.add_argument('--model', default='/Users/tawate/Documents/H2O_Analytics/computer_vision_learning/CV_Engineer_YouTube_Tutorials/03_Image_Detection/detectron2/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml') # specifies the baseline model to use against the training images.

    args = parser.parse_args()

    # Running the below code will initally show the parameters of the selected model
    detectron2_util.train(  args.output_dir,
                            args.data_dir,
                            args.class_list,
                            device=args.device,
                            learning_rate=float(args.learning_rate), # learning rate of training process
                            batch_size=int(args.batch_size),
                            iterations=int(args.iterations), # numer of iterations to run through
                            checkpoint_period=int(args.checkpoint_period), # how often are the model weights saved off
                            model=args.model)