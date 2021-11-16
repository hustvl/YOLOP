from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import os
import json 
import numpy as np

from tqdm import tqdm


def poly2patch(poly2d, closed=False, alpha=1., color=None):
    moves = {'L': Path.LINETO,
             'C': Path.CURVE4}
    points = [p[:2] for p in poly2d]
    codes = [moves[p[2]] for p in poly2d]
    codes[0] = Path.MOVETO

    if closed:
        points.append(points[0])
        codes.append(Path.CLOSEPOLY)


    return mpatches.PathPatch(
        Path(points, codes),
        facecolor=color if closed else 'none',
        edgecolor=color,  # if not closed else 'none',
        lw=1 if closed else 2 * 1, alpha=alpha,
        antialiased=False, snap=True)


def get_areas_v0(objects):
    # print(objects['category'])
    return [o for o in objects
            if 'poly2d' in o and o['category'].startswith('area')]


def draw_drivable(objects, ax):
    plt.draw()

    objects = get_areas_v0(objects)
    for obj in objects:
        if obj['category'] == 'area/drivable':
            color = (1, 1, 1)
        # elif obj['category'] == 'area/alternative':
        #     color = (0, 1, 0)
        else:
            if obj['category'] != 'area/alternative':
                print(obj['category'])
            color = (0, 0, 0)
        # alpha = 0.5
        alpha = 1.0
        poly2d = obj['poly2d']
        ax.add_patch(poly2patch(
                poly2d, closed=True,
                alpha=alpha, color=color))

    ax.axis('off')

def filter_pic(data):
    for obj in data:
        if obj['category'].startswith('area'):
            return True
        else:
            pass
    return False

def main(mode="train"):
    image_dir = "bdd/bdd100k/images/100k/{}".format(mode)
    val_dir = "bdd/bdd100k/labels/100k/{}".format(mode)
    out_dir = 'bdd_seg_gt/{}'.format(mode)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    val_list = os.listdir(val_dir)
    # val_pd = pd.read_json(open(val_json))
    # print(val_pd.head())

    for val_json in tqdm(val_list):
        val_json = os.path.join(val_dir, val_json)
        val_pd = json.load(open(val_json))
        data = val_pd['frames'][0]['objects']
        img_name = val_pd['name']

        remain = filter_pic(data)
        # if remain:
        dpi = 80
        w = 16
        h = 9
        image_width = 1280
        image_height = 720
        fig = plt.figure(figsize=(w, h), dpi=dpi)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
        out_path = os.path.join(out_dir, img_name+'.png')
        ax.set_xlim(0, image_width - 1)
        ax.set_ylim(0, image_height - 1)
        ax.invert_yaxis()
        ax.add_patch(poly2patch(
            [[0, 0, 'L'], [0, image_height - 1, 'L'],
            [image_width - 1, image_height - 1, 'L'],
            [image_width - 1, 0, 'L']],
            closed=True, alpha=1., color=(0, 0, 0)))
        if remain:
            draw_drivable(data, ax)
        fig.savefig(out_path, dpi=dpi)
        plt.close()
    else:
        pass

if __name__ == '__main__':
    main(mode='train')
