import os
import shutil
import yaml

try:
    import cudf as dataframe
except ImportError:
    print("cudf not found, skipping cudf. Use pandas instead")
    import pandas as dataframe

def read_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def read_data(cfg):
    if len(os.listdir(cfg['path']+cfg['labels'])) == 0:
        print("No labels found")
        return None
    elif len(os.listdir(cfg['path']+cfg['labels'])) == 1:
        name = os.listdir(cfg['path']+cfg['labels'])[0]
        filename = cfg['path']+cfg['labels'] +"/"+ name
        return dataframe.read_csv(filename).apply(lambda row: row['image'] + ' ' + \
                                                  row[row == 1].index[0] if 1 in row.values \
                                                  else '', axis=1).to_list()

def move_images_to_folders(data,cfg,output_dir):
    classes = set([x.split(' ')[1] for x in data])
    for class_name in classes:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    for row in data:
        image_name, class_name = row.split(' ')
        src_path = os.path.join(output_dir, image_name + cfg["format_file"])
        dst_path = os.path.join(output_dir, class_name, image_name + cfg["format_file"])
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"Image file {src_path} not found")

if __name__ == "__main__" :
    cfg = read_config("config.yaml")
    data = read_data(cfg)
    move_images_to_folders(data,cfg,cfg["path"]+cfg["images"])
    print("Done")
