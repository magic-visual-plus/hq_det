import os
from copy import copy, deepcopy

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from ultralytics.data.augment import Compose, Format, LetterBox, v8_transforms
from ultralytics.utils.instance import Instances


class HQYOLODataset(Dataset):
    """Adapt an HQ-DET COCO dataset to Ultralytics' augmentation pipeline."""

    def __init__(
        self,
        dataset,
        image_size,
        hyp,
        class_id_map,
        class_id2names,
        augment,
        batch_size,
        ultralytics_format=False,
        rect=False,
        stride=32,
        pad=0.0,
        single_cls=False,
        classes=None,
        fraction=1.0,
        augmentation_strategy=None,
        metadata_mode=False,
    ):
        self.dataset = dataset
        self.imgsz = int(image_size)
        self.hyp = copy(hyp)
        self.class_id_map = {int(k): int(v) for k, v in class_id_map.items()}
        self.class_id2names = {int(k): v for k, v in class_id2names.items()}
        self.augment = bool(augment)
        self.batch_size = int(batch_size)
        self.ultralytics_format = bool(ultralytics_format)
        self.rect = bool(rect)
        self.stride = int(stride)
        self.pad = float(pad)
        self.single_cls = bool(single_cls)
        self.include_classes = (
            None if classes is None else {int(class_id) for class_id in classes}
        )
        self.augmentation_strategy = augmentation_strategy

        self.cache = None
        self.buffer = []
        self.data = {
            "names": self.class_id2names,
            "nc": len(self.class_id2names),
            "flip_idx": [],
            "channels": 3,
        }
        self.use_segments = False
        self.use_keypoints = False
        self.use_obb = False

        self._metadata_mode = bool(
            self.ultralytics_format
            or bool(metadata_mode)
            or self.rect
            or float(fraction) < 1.0
            or self.single_cls
            or self.include_classes is not None
        )
        if self._metadata_mode:
            self._initialize_official_metadata(float(fraction))

        self.max_buffer_length = (
            min(len(self), max(self.batch_size * 8, 1), 1000)
            if self.augment
            else 0
        )
        self.mosaic = bool(
            self.augment and not self.rect and getattr(self.hyp, "mosaic", 0.0)
        )
        self.transforms = self._build_transforms()

    def __len__(self):
        if self._metadata_mode:
            return len(self.labels)
        return len(self.dataset)

    def _initialize_official_metadata(self, fraction):
        if not 0 < fraction <= 1:
            raise ValueError(f"YOLO dataset fraction must be in (0, 1], got {fraction}.")
        if not hasattr(self.dataset, "coco") or not hasattr(self.dataset, "ids"):
            raise TypeError(
                "Metadata-backed HQYOLODataset requires the HQ-DET COCO dataset."
            )

        labels = []
        source_indices = []
        image_ids = []
        for source_index, image_id in enumerate(self.dataset.ids):
            image_info = self.dataset.coco.imgs[int(image_id)]
            height = int(image_info["height"])
            width = int(image_info["width"])
            if height <= 0 or width <= 0:
                raise ValueError(f"Invalid image shape for COCO image {image_id}: {(height, width)}")

            classes = []
            boxes = []
            for annotation in self.dataset.coco.imgToAnns.get(int(image_id), []):
                if int(annotation.get("iscrowd", 0)):
                    continue
                bbox = annotation.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x, y, box_width, box_height = (float(value) for value in bbox)
                x1 = min(max(x, 0.0), float(width))
                y1 = min(max(y, 0.0), float(height))
                x2 = min(max(x + box_width, 0.0), float(width))
                y2 = min(max(y + box_height, 0.0), float(height))
                if x2 <= x1 or y2 <= y1:
                    continue

                original_id = int(annotation["category_id"])
                if original_id not in self.class_id_map:
                    raise ValueError(
                        "Dataset contains a category id missing from the YOLO mapping: "
                        f"{original_id}"
                    )
                class_id = 0 if self.single_cls else self.class_id_map[original_id]
                if self.include_classes is not None and class_id not in self.include_classes:
                    continue

                classes.append([float(class_id)])
                boxes.append(
                    [
                        ((x1 + x2) / 2.0) / width,
                        ((y1 + y2) / 2.0) / height,
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    ]
                )

            cls = np.asarray(classes, dtype=np.float32).reshape(-1, 1)
            bboxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            if len(cls):
                rows = np.concatenate((cls, bboxes), axis=1)
                _, unique_indices = np.unique(rows, axis=0, return_index=True)
                unique_indices = np.sort(unique_indices)
                cls = cls[unique_indices]
                bboxes = bboxes[unique_indices]

            im_file = os.path.abspath(
                os.path.join(self.dataset.root, image_info["file_name"])
            )
            labels.append(
                {
                    "im_file": im_file,
                    "shape": (height, width),
                    "cls": cls,
                    "bboxes": bboxes,
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
            source_indices.append(source_index)
            image_ids.append(int(image_id))

        if fraction < 1.0:
            sample_count = round(len(labels) * fraction)
            labels = labels[:sample_count]
            source_indices = source_indices[:sample_count]
            image_ids = image_ids[:sample_count]
        if not labels:
            raise RuntimeError("No images are available in the YOLO dataset.")

        self.labels = labels
        self._source_indices = source_indices
        self._image_ids = image_ids
        self.im_files = [label["im_file"] for label in labels]
        self.ni = len(labels)
        if self.rect:
            self._set_rectangle()

    def _set_rectangle(self):
        batch_index = np.floor(np.arange(self.ni) / self.batch_size).astype(int)
        num_batches = int(batch_index[-1] + 1)
        shapes = np.asarray([label["shape"] for label in self.labels])
        aspect_ratio = shapes[:, 0] / shapes[:, 1]
        order = aspect_ratio.argsort()

        self.labels = [self.labels[index] for index in order]
        self._source_indices = [self._source_indices[index] for index in order]
        self._image_ids = [self._image_ids[index] for index in order]
        self.im_files = [self.im_files[index] for index in order]
        aspect_ratio = aspect_ratio[order]

        batch_shapes = [[1, 1]] * num_batches
        for batch_id in range(num_batches):
            ratio = aspect_ratio[batch_index == batch_id]
            minimum, maximum = ratio.min(), ratio.max()
            if maximum < 1:
                batch_shapes[batch_id] = [maximum, 1]
            elif minimum > 1:
                batch_shapes[batch_id] = [1, 1 / minimum]

        self.batch_shapes = (
            np.ceil(np.asarray(batch_shapes) * self.imgsz / self.stride + self.pad)
            .astype(int)
            * self.stride
        )
        self.batch = batch_index

    def _build_transforms(self):
        if self.augmentation_strategy is not None:
            return self.augmentation_strategy.build(self)

        if self.augment:
            self.hyp.mosaic = self.hyp.mosaic if not self.rect else 0.0
            self.hyp.mixup = self.hyp.mixup if not self.rect else 0.0
            self.hyp.cutmix = self.hyp.cutmix if not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, self.hyp)
        else:
            transforms = Compose(
                [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
            )

        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=False,
                return_keypoint=False,
                return_obb=False,
                batch_idx=True,
                mask_ratio=getattr(self.hyp, "mask_ratio", 4),
                mask_overlap=getattr(self.hyp, "overlap_mask", True),
                bgr=getattr(self.hyp, "bgr", 0.0) if self.augment else 0.0,
            )
        )
        return transforms

    def close_mosaic(self, hyp=None):
        if self.augmentation_strategy is not None:
            self.transforms = self.augmentation_strategy.close_mosaic(self, hyp)
            return
        if hyp is not None:
            self.hyp = copy(hyp)
        for name in ("mosaic", "copy_paste", "mixup", "cutmix"):
            setattr(self.hyp, name, 0.0)
        self.mosaic = False
        self.transforms = self._build_transforms()

    def _remap_classes(self, classes):
        classes = np.asarray(classes, dtype=np.int64).reshape(-1)
        remapped = np.full(classes.shape, -1, dtype=np.int64)
        for original_id, yolo_id in self.class_id_map.items():
            remapped[classes == original_id] = 0 if self.single_cls else yolo_id

        if (remapped < 0).any():
            unknown = sorted(np.unique(classes[remapped < 0]).tolist())
            raise ValueError(
                f"Dataset contains category ids missing from the YOLO mapping: {unknown}"
            )
        if self.include_classes is not None:
            keep = np.isin(remapped, list(self.include_classes))
            return remapped[keep].astype(np.float32).reshape(-1, 1), keep
        return remapped.astype(np.float32).reshape(-1, 1), None

    def _resize_long_side(self, image, boxes):
        height, width = image.shape[:2]
        ratio = self.imgsz / max(height, width)
        if ratio == 1.0:
            return image, boxes, (height, width)

        new_width = min(int(np.ceil(width * ratio)), self.imgsz)
        new_height = min(int(np.ceil(height * ratio)), self.imgsz)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        boxes = boxes.copy()
        boxes[:, [0, 2]] *= new_width / width
        boxes[:, [1, 3]] *= new_height / height
        return image, boxes, (new_height, new_width)

    def _get_official_image_and_label(self, index):
        label = deepcopy(self.labels[index])
        image_id = self._image_ids[index]
        if hasattr(self.dataset, "_load_image"):
            image = self.dataset._load_image(image_id)
        else:
            image = np.asarray(self.dataset[self._source_indices[index]]["img"])
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {label['im_file']}")

        image = np.asarray(image)
        original_shape = tuple(int(value) for value in image.shape[:2])
        empty_boxes = np.zeros((0, 4), dtype=np.float32)
        image, _, resized_shape = self._resize_long_side(image, empty_boxes)

        if self.augment:
            self.buffer.append(index)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                self.buffer.pop(0)

        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")
        label.pop("shape", None)
        label["img"] = np.ascontiguousarray(image)
        label["image_id"] = int(image_id)
        label["original_shape"] = original_shape
        label["ori_shape"] = original_shape
        label["resized_shape"] = resized_shape
        label["ratio_pad"] = (
            resized_shape[0] / original_shape[0],
            resized_shape[1] / original_shape[1],
        )
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        label["instances"] = Instances(
            bboxes=bboxes,
            segments=(
                np.asarray(segments, dtype=np.float32)
                if len(segments)
                else np.zeros((0, 1000, 2), dtype=np.float32)
            ),
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=normalized,
        )
        return label

    def get_image_and_label(self, index):
        if self._metadata_mode:
            return self._get_official_image_and_label(index)

        source = self.dataset[index]
        image = np.asarray(source["img"])
        boxes = np.asarray(source["bboxes"], dtype=np.float32).reshape(-1, 4)
        classes, keep = self._remap_classes(source["cls"])
        if keep is not None:
            boxes = boxes[keep]

        original_shape = tuple(int(v) for v in image.shape[:2])
        image, boxes, resized_shape = self._resize_long_side(image, boxes)
        image_id = source["image_id"]
        if hasattr(image_id, "item"):
            image_id = image_id.item()

        self.buffer.append(index)
        if len(self.buffer) > self.max_buffer_length:
            self.buffer.pop(0)

        return {
            "img": np.ascontiguousarray(image),
            "im_file": str(image_id),
            "image_id": int(image_id),
            "original_shape": original_shape,
            "ori_shape": original_shape,
            "resized_shape": resized_shape,
            "ratio_pad": (
                resized_shape[0] / original_shape[0],
                resized_shape[1] / original_shape[1],
            ),
            "cls": classes,
            "instances": Instances(
                bboxes=boxes,
                segments=np.zeros((0, 1000, 2), dtype=np.float32),
                keypoints=None,
                bbox_format="xyxy",
                normalized=False,
            ),
        }

    @staticmethod
    def _xywhn_to_xyxy(boxes, height, width):
        xyxy = torch.empty_like(boxes)
        xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * width
        xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * height
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * width
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * height
        xyxy[:, [0, 2]].clamp_(0, width)
        xyxy[:, [1, 3]].clamp_(0, height)
        return xyxy

    def _validate_class_range(self, classes):
        classes = classes.reshape(-1)
        if not classes.numel():
            return
        min_id = int(classes.min().item())
        max_id = int(classes.max().item())
        if min_id < 0 or max_id >= len(self.class_id2names):
            raise ValueError(
                "YOLO class ids must be continuous and inside "
                f"[0, {len(self.class_id2names) - 1}], got [{min_id}, {max_id}]"
            )

    def _finalize_sample(self, sample, index):
        image = sample["img"].float().div_(255.0)
        boxes = sample["bboxes"].float().reshape(-1, 4)
        classes = sample["cls"].long().reshape(-1)
        batch_idx = sample["batch_idx"].long().reshape(-1)
        self._validate_class_range(classes)

        height, width = image.shape[1:]
        boxes_xyxy = self._xywhn_to_xyxy(boxes, height, width)
        image_id = sample.get("image_id", sample.get("im_file", index))
        if hasattr(image_id, "item"):
            image_id = image_id.item()
        if isinstance(image_id, str) and image_id.isdigit():
            image_id = int(image_id)
        elif not isinstance(image_id, (int, np.integer)):
            if (
                    getattr(self, "_metadata_mode", False)
                    and hasattr(self, "_image_ids")
                    and 0 <= int(index) < len(self._image_ids)
            ):
                image_id = self._image_ids[int(index)]
            else:
                image_id = int(index)
                
        original_shape = sample.get(
            "original_shape", sample.get("ori_shape", (height, width))
        )
        return {
            "img": image,
            "cls": classes,
            "bboxes": boxes,
            "bboxes_cxcywh_norm": boxes,
            "bboxes_xyxy": boxes_xyxy,
            "batch_idx": batch_idx,
            "image_id": int(image_id),
            "original_shape": tuple(int(v) for v in original_shape),
        }

    def __getitem__(self, index):
        sample = self.transforms(self.get_image_and_label(index))
        if self.ultralytics_format:
            self._validate_class_range(sample["cls"])
            return sample
        return self._finalize_sample(sample, index)

    @staticmethod
    def collate_fn(batch):
        """Match Ultralytics 8.4.7 YOLODataset.collate_fn."""
        new_batch = {}
        batch = [dict(sorted(sample.items())) for sample in batch]
        keys = batch[0].keys()
        values = list(zip(*[list(sample.values()) for sample in batch]))
        for index, key in enumerate(keys):
            value = values[index]
            if key in {"img", "text_feats", "sem_masks"}:
                value = torch.stack(value, 0)
            elif key == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if key in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[key] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for index in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][index] += index
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


__all__ = ["HQYOLODataset"]
