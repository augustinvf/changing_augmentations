from typing import Optional, Callable, Tuple, Any

from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as T

from PIL import Image

class TrainDatasetTransformsCIFAR10(CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        self_supervised_augmentations = None,
    ) -> None:
        
        super().__init__(root, train, transform, target_transform, download)
        self.self_supervised_augmentations = self_supervised_augmentations

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

        if self.self_supervised_augmentations:
            img = self.self_supervised_augmentations(img, target)
        elif self.transform :
            img = self.transform(img)

        if self.target_transform :
            target = self.target_transform(target)

        return (img, target)

    def update_self_supervised_augmentations(self, self_supervised_augmentations):
        self.self_supervised_augmentations = self_supervised_augmentations
