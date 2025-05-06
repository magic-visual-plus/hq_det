import sys
import ultralytics.nn.tasks


if __name__ == '__main__':
    model, weights = ultralytics.nn.tasks.attempt_load_one_weight(sys.argv[1])
    print(model)
    pass