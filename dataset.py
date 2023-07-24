from typing import Optional, Callable, Tuple, Any

from torchvision.datasets.cifar import CIFAR10
from PIL import Image

class DatasetTransformsCIFAR10(CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        class_transform = None,
        power_list = [],
        current_operations = [],
    ) -> None:
        
        super().__init__(root, train, transform, target_transform, download)
        self.class_transform = class_transform
        self.power_list = power_list
        self.current_operations = current_operations

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.class_transform:
            assert self.power_list != [], "Power list shouldn't be empty if you want to change augmentations"
            new_transform = self.class_transform(self.power_list, target)
            img = new_transform(img)

        elif self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

    def update_transform(self, class_transform: type):
        self.class_transform=class_transform

    def update_power_list(self, power_list: list):
        self.power_list = power_list

    def update_current_operations(self, current_operations: list):
        self.current_operations = current_operations
