
import cv2
import torch
import torchvision.transforms.functional as VF
import ultralytics.utils.ops

if __name__ == '__main__':
    import sys
    from ultralytics import YOLO

    # Load a model
    model = YOLO(sys.argv[1])  # load an official detection model

    img = cv2.imread(sys.argv[2])
    img = cv2.resize(img, (800, 800))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgt = VF.to_tensor(imgt)
    # imgt = VF.normalize(imgt, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    imgt = imgt.unsqueeze(0)
    imgt = torch.tile(imgt, (2, 1, 1, 1))
    imgt[0, :, :, :] = 0
    imgt = imgt.to('cuda:0')

    print(imgt)
    # Predict with the model
    results = model.predict(source=img, show=True, save=True, save_txt=True, conf=0.5, iou=0.5)
    print(results[0].boxes)


    forward_result = model.model(imgt)

    preds = ultralytics.utils.ops.non_max_suppression(
            forward_result,
            0.01,
            0.01,
            None,
        )
    print(preds)