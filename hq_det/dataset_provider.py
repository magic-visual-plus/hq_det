import os
from .dataset import CocoDetection, CombinedDataset


class HQDatasetProvider(object):
    def __init__(self, ):
        pass

    def build_train_dataset(self, transforms):
        raise NotImplementedError()
    
    def build_valid_dataset(self, transforms):
        raise NotImplementedError()
    pass



class HQDatasetProviderRoboflow(HQDatasetProvider):
    def __init__(self, data_root):
        super().__init__()
        if not (isinstance(data_root, str) and len(data_root) > 0):
            raise ValueError(f"Invalid data_root: {data_root}")

        self.data_root = data_root
        pass

    def build_dataset(self, path, transforms):
        annotation_file = os.path.join(path, "_annotations.coco.json")
        dataset = CocoDetection(
            path, annotation_file, transforms=transforms
        )
        return dataset
    
    def build_train_dataset(self, train_transforms):
        return self.build_dataset(os.path.join(self.data_root, "train"), train_transforms)
    
    def build_valid_dataset(self, val_transforms):
        return self.build_dataset(os.path.join(self.data_root, "valid"), val_transforms)
    pass


class HQDatasetProviderCombined(HQDatasetProvider):
    def __init__(self, data_root_list, data_weight_list, valid_path):
        super().__init__()
        self.data_root_list = data_root_list
        self.data_weight_list = data_weight_list
        self.valid_path = valid_path
        pass

    def build_train_dataset(self, train_transforms):
        datasets = []
        for path in self.data_root_list:
            dataset = CocoDetection(path, os.path.join(path, "_annotations.coco.json"), transforms=train_transforms)
            datasets.append(dataset)
            pass
        
        combined_dataset = CombinedDataset(datasets, self.data_weight_list)
        return combined_dataset
    
    def build_valid_dataset(self, val_transforms):
        return CocoDetection(self.valid_path, os.path.join(self.valid_path, "_annotations.coco.json"), transforms=val_transforms)
    pass