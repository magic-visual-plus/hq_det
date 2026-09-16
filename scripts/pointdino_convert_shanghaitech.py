"""Convert ShanghaiTech .mat points to MMEngine's BaseDetDataset JSON.

Point coordinates are copied exactly, without a one-based or center offset,
to preserve the source PointDINO annotation convention.
"""

import argparse
import json
import re
from pathlib import Path

import scipy.io as sio
from PIL import Image


def natural_key(name):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r'(\d+)', name)]


def convert(img_dir, gt_dir, out):
    img_dir, gt_dir, out = Path(img_dir), Path(gt_dir), Path(out)
    image_files = sorted(
        (path for path in img_dir.iterdir()
         if path.suffix.lower() in ('.jpg', '.jpeg', '.png')),
        key=lambda path: natural_key(path.name))
    data_list = []
    for img_id, img_path in enumerate(image_files):
        gt_path = gt_dir / f'GT_{img_path.stem}.mat'
        if not gt_path.is_file():
            raise FileNotFoundError(f'Cannot find annotation: {gt_path}')
        mat = sio.loadmat(gt_path)
        if 'image_info' not in mat:
            raise KeyError(f'image_info not found in {gt_path}')
        points = mat['image_info'][0, 0][0, 0][0]
        with Image.open(img_path) as img:
            width, height = img.size
        instances = [dict(point=[float(point[0]), float(point[1])],
                          point_label=0, ignore_flag=0) for point in points]
        data_list.append(dict(img_id=img_id, img_path=img_path.name,
                              width=width, height=height, instances=instances))

    output = dict(metainfo=dict(classes=['point']), data_list=data_list)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as stream:
        json.dump(output, stream)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--gt-dir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = convert(args.img_dir, args.gt_dir, args.out)
    print('images:', len(output['data_list']))
    print('points:', sum(len(item['instances']) for item in output['data_list']))
    print('saved:', args.out)


if __name__ == '__main__':
    main()
