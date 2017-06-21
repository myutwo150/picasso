import os
from PIL import Image
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

IMG_DIM = (299, 299, 3)

mapping = {}
base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, 'data-volume', 'index.csv')) as f_in:
    f_in.readline()
    for line in f_in:
        index, name = line.strip().split(',')
        mapping[int(index)] = name


def padcrop(img, target_size=299):
    ratio = target_size / img.height
    img = img.resize((int(round(img.width * ratio)), target_size), resample=Image.BILINEAR)
    if img.width >= img.height:
        # landscape image, do central crop
        left = int(round(img.width / 2 - target_size / 2))
        return img.crop((left, 0, left + target_size, target_size))
    else:
        # portrait image, repeat left and right margins
        left_margin = img.crop((0, 0, 1, img.height))
        right_margin = img.crop((img.width - 1, 0, img.width, img.height))
        new_img = Image.new('RGB', (target_size, target_size))
        for i in range(target_size // 2):
            new_img.paste(left_margin, (i, 0))
            new_img.paste(right_margin, (target_size - i - 1, 0))
        new_img.paste(img, (int(round((target_size - img.width) / 2)), 0))
        return new_img


def preprocess(targets):
    image_arrays = []
    for target in targets:
        arr = image.img_to_array(padcrop(target))
        image_arrays.append(arr)

    all_targets = np.array(image_arrays)
    return preprocess_input(all_targets)


def postprocess(output_arr):
    images = []
    for row in output_arr:
        im_array = row.reshape(IMG_DIM[:2])
        images.append(im_array)

    return images


def prob_decode(probability_array, top=5):
    ret = []
    for row in probability_array:
        prob_result = []
        indices = np.argsort(row)[::-1]
        for i in range(top):
            idx = indices[i]
            prob_result.append({
                'index': idx,
                'name': mapping[idx],
                'prob': '{:.4f}'.format(row[idx])
            })
        ret.append(prob_result)
    return ret
