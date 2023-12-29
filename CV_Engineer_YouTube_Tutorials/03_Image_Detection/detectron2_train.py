import argparse

import detectron2_util

DATA_PATH = '/Users/tawate/Documents/H2O_Analytics/data/cv_engineer_youtube_data'

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
    parser.add_argument('--class-list', default='./detectron2_class.names')
    parser.add_argument('--data-dir', default=DATA_PATH + '/data')
    parser.add_argument('--output-dir', default='./output')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--learning-rate', default=0.00025)
    parser.add_argument('--batch-size', default=4)
    parser.add_argument('--iterations', default=10000)
    parser.add_argument('--checkpoint-period', default=500)
    parser.add_argument('--model', default='COCO-Detection/retinanet_R_101_FPN_3x.yaml') # specifies the baseline model to use against the training images.

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
    
    
    # Output Directory will show all the checkpoint model weights in an individual file per checkpoint.
    # 