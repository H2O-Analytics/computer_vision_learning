import os

import boto3
import cv2

import credentials

DATA_PATH = '/Users/tawate/Documents/H2O_Analytics/data/cv_engineer_youtube_data'
output_dir = DATA_PATH + '/output'
output_dir_imgs = os.path.join(output_dir, 'imgs')
output_dir_anns = os.path.join(output_dir, 'anns')

# create AWS Reko client. Region must be specified.
reko_client = boto3.client('rekognition',
                           region_name = 'us-east-1',
                           aws_access_key_id=credentials.access_key,
                           aws_secret_access_key=credentials.secret_key)

# set the target class
target_class = 'Zebra'

# load video
cap = cv2.VideoCapture(DATA_PATH + '/zebras.mp4')

frame_nmr = -1

# read frames
ret = True
while ret:
    ret, frame = cap.read()

    if ret:

        frame_nmr += 1
        H, W, _ = frame.shape

        # convert frame to jpg
        _, buffer = cv2.imencode('.jpg', frame)

        # convert buffer to bytes
        image_bytes = buffer.tobytes()

        # detect objects using aws rekognition: credentials file must be named credetials, or a CredentialNotFound Error will appear.
        response = reko_client.detect_labels(Image={'Bytes': image_bytes},
                                             MinConfidence=50)
        
        # opens a text file and writes the locations and size of each bounding box instance for each frame. zfill(6) gives the txt files all the same length for organization purposes.
        with open(os.path.join(output_dir_anns, 'frame_{}.txt'.format(str(frame_nmr).zfill(6))), 'w') as f:
            for label in response['Labels']:
                if label['Name'] == target_class:
                    for instance_nmr in range(len(label['Instances'])):
                        # creates the bounding box object for each instance of zebra in each frame. Uses the yolo format. 
                        bbox = label['Instances'][instance_nmr]['BoundingBox'] 
                        x1 = bbox['Left'] #left side of bounding box
                        y1 = bbox['Top'] #top of bounding box
                        width = bbox['Width'] #width of bounding box
                        height = bbox['Height'] #height of bounding box

                        # write detections
                        f.write('{} {} {} {} {}\n'.format(0, #always number zero because 0 index and we are only detecting one object
                                                          (x1 + width / 2),
                                                          (y1 + height / 2),
                                                          width,
                                                          height)
                                )
            # clost text file for each frame
            f.close()

        # write image file for each frame with correpsonding frame number. Each annotation txt file and frame image file with have the same number association
        cv2.imwrite(os.path.join(output_dir_imgs, 'frame_{}.jpg'.format(str(frame_nmr).zfill(6))), frame)
        print('frame_{}.jpg'.format(str(frame_nmr).zfill(6)))