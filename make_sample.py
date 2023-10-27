import random
import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

def init_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
    
# 输出文件夹
sample_dir = 'kitchen'
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)
    
dataset_config = {
    'train': {
        'dir': 'train',
        'num': 700
    },
    'valid': {
        'dir': 'valid',
        'num': 200
    }
}

for _, value in dataset_config.items():
    init_dir(os.path.join(sample_dir, value['dir']))
    init_dir(os.path.join(sample_dir, value['dir'], 'images'))
    init_dir(os.path.join(sample_dir, value['dir'], 'labels'))

back_config = {
    'img_dir': 'images/background',
    'size': 800
}

object_config = {
    'rat': {
        'class': 0,
        'img_dir': 'images/rat2',
        'max_num_per_img': 4,      # 目标在背景图片中的最大个数
        'ratio_range': [0.05, 0.08], # 目标尺寸在背景图片中的比率范围
        'rotate_p': 0.3
    },
    'cockroach': {
        'class': 1,
        'img_dir': 'images/cockroach2',
        'max_num_per_img': 10,      # 目标在背景图片中的最大个数
        'ratio_range': [0.03, 0.08], # 目标尺寸在背景图片中的比率范围
        'rotate_p': 0.3
    }
}

def get_images(image_dir, crop_size = None):
    images = []
    for filename in os.listdir(image_dir):
        if filename.startswith('.'):
            continue
        filepath = os.path.join(image_dir, filename)
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if crop_size:
            image = crop_img(image, crop_size)
        images.append(image)
    return images

def crop_img(img, size):
    h = img.shape[0]
    w = img.shape[1]
    min_size = min(w, h)
    ratio = min_size / size
    new_w = int(w / ratio)
    new_h = int(h / ratio)
    img = cv2.resize(img, (new_w, new_h))
    row0 = (new_h - size) // 2 + 1
    row1 = (new_h + size) // 2 - 1
    col0 = (new_w - size) // 2 + 1
    col1 = (new_w + size) // 2 - 1
    img = img[row0:row1, col0:col1]
    return img

# 读取所有背景图片和目标图片

back_images = get_images(back_config['img_dir'], back_config['size'])

object_image_dict = {}
for key, value in object_config.items():
    object_image_dict[key] = get_images(value['img_dir'])
    
# 随机选择一张背景图片
def get_background():
    index = random.randint(0, len(back_images)-1)
    return back_images[index]

# 随机选择多张目标图片

def get_objects():
    object_dict = {}
    for key, value in object_image_dict.items():
        max_num = object_config[key]['max_num_per_img']
        num = random.randint(0, max_num)
        if num > 0:
            object_dict[key] = random.choices(value, k = num)
        else:
            object_dict[key] = []
    return object_dict

def get_object_size(object_key, object_shape, back_shape):
    ratio_range = object_config[object_key]['ratio_range']
    ratio = random.uniform(ratio_range[0], ratio_range[1])
    h_ratio = object_shape[0]/back_shape[0]
    w_ratio = object_shape[1]/back_shape[1]
    max_ratio = max(h_ratio, w_ratio)
    r = ratio / max_ratio
    return int(object_shape[1] * r), int(object_shape[0] * r)

def get_yolo_rect(img, key, rect):
    img_w = img.shape[1]
    img_h = img.shape[0]
    center_x = (rect[0] + rect[2]) // 2
    center_y = (rect[1] + rect[3]) // 2
    rect_w = rect[2] - rect[0]
    rect_h = rect[3] - rect[1]
    
    cls = object_config[key]['class']
    return f'{cls} {center_x/img_w} {center_y/img_h} {rect_w/img_w} {rect_h/img_h}'
    
def save_sample(sample_dir, img, index, rect_dict):
    
    lines = []
    for key, value in rect_dict.items():
        for rect in value:
            # img = cv2.rectangle(img, (rect[0],rect[1]), (rect[2],rect[3]), (0, 0, 255), 2)
            line = get_yolo_rect(img, key, rect)
            lines.append(line)
            
    filename = f'{index:0>6}'
    img_dir = os.path.join(sample_dir, 'images')
    img_path = os.path.join(img_dir, f'{filename}.jpg')
    cv2.imwrite(img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    
    lb_dir = os.path.join(sample_dir, 'labels')
    lb_path = os.path.join(lb_dir, f'{filename}.txt')
    with open(lb_path, 'w') as f:
        for line in lines:
            f.write(line+'\n')
            

def add_image(src, part, point):
    src_height, src_width, channel_num = src.shape
    height, width = part.shape[:2]
    if point[0] + width >= src_width:
        width = width - point[0] - 1
    if point[1] + height >= src_height:
        height = src_height - point[1] - 1

    roi = src[point[1]:point[1]+height, point[0]:point[0]+width]
    if channel_num == 4:
        channels = cv2.split(part)
        cv2.copyTo(part, channels[3], roi)
    else:
        cv2.copyTo(part, None, roi)
    del roi

def attach(back_img, img, point):
    x = point[0]
    y = point[1]
    w = img.shape[1]
    h = img.shape[0]

    part = back_img[y:h+y, x:w+x]
 
    # 获取前景图像的 alpha 通道（如果有的话）
    alpha = img[:, :, 3] / 255.0
    
    # 分离前景和背景的颜色通道
    fore_img = img[:, :, 0:3]
    part = part[:, :, 0:3]
    # 扩展 alpha 通道的维度，以便进行乘法运算
    alpha = np.expand_dims(alpha, axis=2)
    # 根据 alpha 通道对前景图像和背景图像进行混合
    blended = (part * (1 - alpha) + fore_img * alpha).astype(np.uint8)

    add_image(back_img, blended, (x, y))

def rotate(key, img):
    rotate_p = object_config[key]['rotate_p']
    p = int(rotate_p*1000)
    num = random.randint(0, p-1)
    if num > p//2:
        return img
    level = p//3
    if num < level:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif num < level * 2:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        img = cv2.rotate(img, cv2.ROTATE_180)
    return img

def make_sample():
    back_img = get_background()
    back_img = back_img.copy()
    object_dict = get_objects()
    
    flag = False
    # 保存方框 key: 类型， value: 二维数组，[n, 4]
    rect_dict = {}
    for key, value in object_dict.items():
        if len(value) == 0:
            continue
        rects = []
        flag = True
        for img in value:
            img = rotate(key, img)
            img_size = get_object_size(key, img.shape, back_img.shape)
            img = cv2.resize(img, img_size)
            
            img_x = random.randint(1, back_img.shape[1] - img_size[0])
            img_y = random.randint(1, back_img.shape[0] - img_size[1])
        
            attach(back_img, img, (img_x, img_y))
            
            rects.append((img_x, img_y, img_x + img_size[0], img_y + img_size[1]))
        rect_dict[key] = rects
            
    # 保存结果
    if flag:
        return back_img, rect_dict
    else:
        return None
    
# 开始生成样本

# train
print('开始生成train数据集...')
train_dir = os.path.join(sample_dir, dataset_config['train']['dir'])
for i in tqdm(range(dataset_config['train']['num'])):
    result = make_sample()
    if result == None:
        continue
    save_sample(train_dir, result[0], i, result[1])
print('train数据集生成完毕')
    
# valid
print('开始生成valid数据集...')
valid_dir = os.path.join(sample_dir, dataset_config['valid']['dir'])
for i in tqdm(range(dataset_config['valid']['num'])):
    result = make_sample()
    if result == None:
        continue
    save_sample(valid_dir, result[0], i, result[1])
print('valid数据集生成完毕')