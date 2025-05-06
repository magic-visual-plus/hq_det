import sys
from hq_det import split_utils


if __name__ == '__main__':

    split_utils.split_coco(sys.argv[1], sys.argv[2], 1024, 20, 2)
    pass