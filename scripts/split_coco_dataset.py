import sys
from hq_det import split_utils


if __name__ == '__main__':

    split_utils.split_coco(sys.argv[1], sys.argv[2], 2048, 0, 100, add_global=False, box_area_thr=0.0)
    pass