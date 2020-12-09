import skimage.exposure
from skimage.color import rgb2gray
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

TEMPLATES_DIR = 'templates'
UNSORTED_DIR = 'unsorted'


def get_scale(img):
    if img.ndim == 2:
        return 'grayscale'
    return 'color'


def get_main_color(img):
    rgb_tuple = ['red', 'green', 'blue']
    red_hist = skimage.exposure.histogram(img[:, :, 0])
    green_hist = skimage.exposure.histogram(img[:, :, 1])
    blue_hist = skimage.exposure.histogram(img[:, :, 2])

    intensity = np.array([(x * y).sum() for x, y in (hist for hist in (red_hist, green_hist, blue_hist))])
    if not all(intensity[2] == intensity[x] for x in range(2)):
        return rgb_tuple[intensity.argmax()]
    return 'none'


def get_gray_index(img):
    img_gray = rgb2gray(img)
    hist_grey = skimage.exposure.histogram(img_gray, normalize=True)
    return int(hist_grey[0].std() * 10 ** 9)


def hist_compare(*args):
    arrays = tuple([np.zeros(256) for _ in range(2)])
    for h, ar in zip(args, arrays):
        for i, bn in enumerate(h[1]):
            ar[bn] = h[0][i]
    diff_array = abs(arrays[0] - arrays[1])
    return diff_array.sum()


images_list = os.listdir(TEMPLATES_DIR)
scales = []
main_colors = []
grey_indexes = []

for image in images_list:
    pic_color = imageio.imread(f'{TEMPLATES_DIR}/{image}')

    scale = get_scale(pic_color)
    main_color = get_main_color(pic_color)
    grey_index = get_gray_index(pic_color)

    scales.append(scale)
    main_colors.append(main_color)
    grey_indexes.append(grey_index)

multi_index = pd.MultiIndex.from_arrays([scales, main_colors, grey_indexes], names=['scale', 'main_color', 'gray_std'])
cache_index = pd.Series(data=images_list, index=multi_index, name='file_name')

color_ignore = True

unsorted_images = os.listdir(UNSORTED_DIR)
for image in unsorted_images:
    image_to_sort = imageio.imread(f'{UNSORTED_DIR}/{image}')

    curr_scale = get_scale(image_to_sort)
    curr_grey_index = get_gray_index(image_to_sort)

    if color_ignore:
        gray_index_array = np.array(cache_index.index.levels[2])
        file_names = cache_index.to_list()
    else:
        curr_main_color = get_main_color(image_to_sort)
        gray_index_array = np.array(cache_index['color'][curr_main_color].index)
        file_names = cache_index['color'][curr_main_color].to_list()
    data = zip(abs(gray_index_array - curr_grey_index), file_names)
    diff_std = pd.DataFrame(data=data, index=list(gray_index_array), columns=['diff_std', 'file_name'])

    diff_std.sort_values(by='diff_std', inplace=True)

    sample = diff_std[:min(20, len(diff_std))]
    hist_diffs = []
    sample_file_names = sample['file_name']
    sample_indexes = sample.index
    for f in sample_file_names:
        image_file = imageio.imread(f'{TEMPLATES_DIR}/{f}')
        hist_file = skimage.exposure.histogram(image_file, normalize=True)
        hist_grey = skimage.exposure.histogram(image_to_sort, nbins=256, normalize=True)
        hist_diff = hist_compare(hist_grey, hist_file)
        hist_diffs.append(hist_diff)

    data = zip(hist_diffs, sample_file_names)
    diff = pd.DataFrame(data=data, index=sample_indexes, columns=['diff_hist', 'file_name'])
    diff.sort_values(by='diff_hist', inplace=True)

    # visualization
    file_names_list = diff['file_name'][:min(3, len(diff))].to_list()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(8, 3))
    for aa in (ax1, ax2, ax3, ax4):
        aa.set_axis_off()
    ax1.imshow(imageio.imread(f'{UNSORTED_DIR}/{image}'))
    ax1.set_title('Sorting')
    try:
        ax2.imshow(imageio.imread(f"{TEMPLATES_DIR}/{file_names_list[0]}"))
        ax2.set_title('First')
        ax3.imshow(imageio.imread(f"{TEMPLATES_DIR}/{file_names_list[1]}"))
        ax3.set_title('Second')
        ax4.imshow(imageio.imread(f"{TEMPLATES_DIR}/{file_names_list[2]}"))
        ax4.set_title('Third')
    except IndexError:
        pass

    plt.tight_layout()
    plt.show()
