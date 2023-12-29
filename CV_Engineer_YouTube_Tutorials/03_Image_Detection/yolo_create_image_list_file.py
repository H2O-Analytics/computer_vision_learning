import os


alpaca_id = '/m/0pcr'
DATA_PATH = '/Users/tawate/Documents/H2O_Analytics/data/cv_engineer_youtube_data/object_detection/'
train_bboxes_filename = DATA_PATH + 'oidv6-train-annotations-bbox.csv'
validation_bboxes_filename = DATA_PATH + 'validation-annotations-bbox.csv'
test_bboxes_filename = DATA_PATH + 'test-annotations-bbox.csv'

image_list_file_path = DATA_PATH + 'image_list_file'

image_list_file_list = []
for j, filename in enumerate([train_bboxes_filename, validation_bboxes_filename, test_bboxes_filename]):
    print(filename)
    with open(filename, 'r') as f:
        line = f.readline()
        while len(line) != 0:
            id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(',')[:13]
            if class_name in [alpaca_id] and id not in image_list_file_list:
                image_list_file_list.append(id)
                with open(image_list_file_path, 'a') as fw:
                    fw.write('{}/{}\n'.format(['train', 'validation', 'test'][j], id))
            line = f.readline()

        f.close()