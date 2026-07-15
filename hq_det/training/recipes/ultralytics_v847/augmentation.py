"""Frozen 8.4.7 detection augmentation composition."""

from copy import copy

from ultralytics.data.augment import (
    Albumentations,
    Compose,
    CopyPaste,
    CutMix,
    Format,
    LetterBox,
    MixUp,
    Mosaic,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)

from hq_det.training.interfaces import AugmentationStrategy


class UltralyticsV847Augmentation(AugmentationStrategy):
    """Own the transform graph while reusing pinned low-level image primitives."""

    def build(self, dataset):
        hyp = dataset.hyp
        if dataset.augment:
            hyp.mosaic = hyp.mosaic if not dataset.rect else 0.0
            hyp.mixup = hyp.mixup if not dataset.rect else 0.0
            hyp.cutmix = hyp.cutmix if not dataset.rect else 0.0
            transforms = self._build_train(dataset, hyp)
        else:
            transforms = Compose(
                [LetterBox(new_shape=(dataset.imgsz, dataset.imgsz), scaleup=False)]
            )

        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=dataset.use_segments,
                return_keypoint=dataset.use_keypoints,
                return_obb=dataset.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if dataset.augment else 0.0,
            )
        )
        return transforms

    def _build_train(self, dataset, hyp):
        imgsz = dataset.imgsz
        mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
        affine = RandomPerspective(
            degrees=hyp.degrees,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
            pre_transform=LetterBox(new_shape=(imgsz, imgsz)),
        )
        pre_transform = Compose([mosaic, affine])
        if hyp.copy_paste_mode == "flip":
            pre_transform.insert(
                1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode)
            )
        else:
            pre_transform.append(
                CopyPaste(
                    dataset,
                    pre_transform=Compose(
                        [Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]
                    ),
                    p=hyp.copy_paste,
                    mode=hyp.copy_paste_mode,
                )
            )

        flip_idx = dataset.data.get("flip_idx", [])
        if dataset.use_keypoints:
            keypoint_shape = dataset.data.get("kpt_shape")
            if not flip_idx and (hyp.fliplr > 0.0 or hyp.flipud > 0.0):
                hyp.fliplr = hyp.flipud = 0.0
            elif flip_idx and len(flip_idx) != keypoint_shape[0]:
                raise ValueError(
                    "flip_idx length must match the number of keypoints."
                )

        return Compose(
            [
                pre_transform,
                MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
                CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
                Albumentations(
                    p=1.0, transforms=getattr(hyp, "augmentations", None)
                ),
                RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
                RandomFlip(
                    direction="vertical", p=hyp.flipud, flip_idx=flip_idx
                ),
                RandomFlip(
                    direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx
                ),
            ]
        )

    def close_mosaic(self, dataset, hyp=None):
        if hyp is not None:
            dataset.hyp = copy(hyp)
        for name in ("mosaic", "copy_paste", "mixup", "cutmix"):
            setattr(dataset.hyp, name, 0.0)
        dataset.mosaic = False
        return self.build(dataset)


__all__ = ["UltralyticsV847Augmentation"]
