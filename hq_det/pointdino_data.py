"""Read COCO point annotations without importing model or training libraries."""

from copy import deepcopy
import json
import math
from numbers import Integral, Real


def _identifier(value, location):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{location} must be an integer.')
    return int(value)


def _classes(names, location):
    if (not isinstance(names, (list, tuple)) or not names
            or not all(isinstance(name, str) and name for name in names)):
        raise ValueError(f'{location} must contain nonempty class names.')
    # Category IDs identify classes; different IDs may share a display name.
    return tuple(names)


def _category_ids(values, location):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f'{location} must contain category IDs.')
    values = tuple(_identifier(value, location) for value in values)
    if len(set(values)) != len(values):
        raise ValueError(f'{location} contains duplicate category IDs.')
    return values


def _point_instance(instance, label, location):
    if 'point' not in instance:
        raise ValueError(
            f'{location} is missing point=[x, y]. Bounding boxes are not '
            'automatically converted to point annotations.')
    point = instance['point']
    if (not isinstance(point, (list, tuple)) or len(point) != 2
            or any(isinstance(value, bool) or not isinstance(value, Real)
                   or not math.isfinite(value) for value in point)):
        raise ValueError(f'{location}.point must contain two finite coordinates.')
    ignore = instance.get('ignore_flag', instance.get('iscrowd', 0))
    if not isinstance(ignore, Integral) or ignore not in (0, 1):
        raise ValueError(f'{location}.ignore_flag/iscrowd must be 0 or 1.')
    return dict(point=list(point), point_label=label, ignore_flag=int(ignore))


def _validate_image(record, location):
    if not isinstance(record.get('img_path'), str) or not record['img_path']:
        raise ValueError(f'{location}.img_path/file_name must be a nonempty string.')
    for key in ('width', 'height'):
        if key in record:
            value = _identifier(record[key], f'{location}.{key}')
            if value <= 0:
                raise ValueError(f'{location}.{key} must be positive.')


def read_point_annotations(path, class_names=None, category_ids=None):
    """Normalize a COCO-point or legacy PointDINO annotation file.

    Returns a dict with ``format``, ``metainfo`` and ``data_list``. The latter
    retains original pixel coordinates and uses contiguous ``point_label``
    values. For COCO, labels follow sorted category IDs, including categories
    with duplicate display names. ``metainfo.category_ids[label]`` recovers
    the original ID; ``metainfo.classes[label]`` gives its name.

    Pass both metadata sequences from training to validate another split's
    ID-to-name mapping. Missing point annotations are errors, even if a bbox
    is present. No image files or annotations are modified by this function.
    """
    with open(path, encoding='utf-8') as stream:
        content = json.load(stream)
    if not isinstance(content, dict):
        raise ValueError(f'{path}: annotation JSON must be an object.')

    if any(key in content for key in ('images', 'annotations', 'categories')):
        for key in ('images', 'annotations', 'categories'):
            if not isinstance(content.get(key), list):
                raise ValueError(f'{path}: COCO-point requires a {key} list.')
        categories = {}
        for index, category in enumerate(content['categories']):
            location = f'{path}: categories[{index}]'
            if not isinstance(category, dict):
                raise ValueError(f'{location} must be an object.')
            category_id = _identifier(category.get('id'), f'{location}.id')
            if category_id in categories:
                raise ValueError(f'{location}: duplicate category ID {category_id}.')
            categories[category_id] = category.get('name')
        resolved_ids = tuple(sorted(categories))
        resolved_names = _classes(
            [categories[key] for key in resolved_ids], f'{path}: category names')
        label_by_category = {key: index for index, key in enumerate(resolved_ids)}
        records = []
        by_image = {}
        for index, image in enumerate(content['images']):
            location = f'{path}: images[{index}]'
            if not isinstance(image, dict):
                raise ValueError(f'{location} must be an object.')
            image_id = _identifier(image.get('id'), f'{location}.id')
            if image_id in by_image:
                raise ValueError(f'{location}: duplicate image ID {image_id}.')
            record = deepcopy(image)
            record.update(img_id=image_id, img_path=image.get('file_name'), instances=[])
            _validate_image(record, location)
            records.append(record)
            by_image[image_id] = record
        seen_annotations = set()
        for index, annotation in enumerate(content['annotations']):
            location = f'{path}: annotations[{index}]'
            if not isinstance(annotation, dict):
                raise ValueError(f'{location} must be an object.')
            annotation_id = _identifier(annotation.get('id'), f'{location}.id')
            location += f' (id={annotation_id})'
            if annotation_id in seen_annotations:
                raise ValueError(f'{location}: duplicate annotation ID.')
            seen_annotations.add(annotation_id)
            image_id = _identifier(annotation.get('image_id'), f'{location}.image_id')
            if image_id not in by_image:
                raise ValueError(f'{location}: unknown image_id {image_id}.')
            category_id = _identifier(
                annotation.get('category_id'), f'{location}.category_id')
            if category_id not in label_by_category:
                raise ValueError(f'{location}: unknown category_id {category_id}.')
            instance = _point_instance(
                annotation, label_by_category[category_id], location)
            by_image[image_id]['instances'].append(instance)
        metainfo = dict(classes=resolved_names, category_ids=resolved_ids)
        annotation_format = 'coco_point'
    else:
        if (not isinstance(content.get('metainfo'), dict)
                or not isinstance(content.get('data_list'), list)):
            raise ValueError(
                f'{path}: expected COCO images/annotations/categories or '
                'legacy metainfo/data_list.')
        metainfo = deepcopy(content['metainfo'])
        resolved_names = _classes(
            metainfo.get('classes', class_names), f'{path}: metainfo.classes')
        resolved_ids = _category_ids(
            metainfo.get('category_ids', list(range(len(resolved_names)))),
            f'{path}: metainfo.category_ids')
        if len(resolved_ids) != len(resolved_names):
            raise ValueError(f'{path}: category_ids and classes lengths differ.')
        records = deepcopy(content['data_list'])
        seen_images = set()
        for index, record in enumerate(records):
            location = f'{path}: data_list[{index}]'
            if not isinstance(record, dict):
                raise ValueError(f'{location} must be an object.')
            image_id = _identifier(record.setdefault('img_id', index), f'{location}.img_id')
            if image_id in seen_images:
                raise ValueError(f'{location}: duplicate img_id {image_id}.')
            seen_images.add(image_id)
            _validate_image(record, location)
            instances = record.get('instances', [])
            if not isinstance(instances, list):
                raise ValueError(f'{location}.instances must be a list.')
            normalized = []
            for instance_index, instance in enumerate(instances):
                point_location = f'{location}.instances[{instance_index}]'
                if not isinstance(instance, dict):
                    raise ValueError(f'{point_location} must be an object.')
                label = _identifier(instance.get('point_label'), f'{point_location}.point_label')
                if not 0 <= label < len(resolved_names):
                    raise ValueError(f'{point_location}: point_label {label} is out of range.')
                normalized.append(_point_instance(instance, label, point_location))
            record['instances'] = normalized
        metainfo.update(classes=resolved_names, category_ids=resolved_ids)
        annotation_format = 'legacy'

    if category_ids is not None:
        expected_ids = _category_ids(category_ids, 'configured category_ids')
        if expected_ids != resolved_ids:
            raise ValueError(
                f'{path}: category IDs/order {resolved_ids} differ from '
                f'training/config {expected_ids}.')
    if class_names is not None:
        expected_names = _classes(class_names, 'configured class_names')
        if expected_names != resolved_names:
            raise ValueError(
                f'{path}: category ID-to-name mapping differs from training/config.')
    return dict(format=annotation_format, metainfo=metainfo, data_list=records)
