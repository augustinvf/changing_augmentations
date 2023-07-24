from typing import Optional, Callable, Tuple, Any

from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as T

from PIL import Image

test_transformation = T.Compose(
    [
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2470, 0.2435, 0.2616])
    ]
)

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
        operation_list = [],
        test_transformation = [],
    ) -> None:
        
        super().__init__(root, train, transform, target_transform, download)
        self.class_transform = class_transform
        self.power_list = power_list
        self.operation_list = operation_list
        self.test_transformation = test_transformation

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
            new_transform = self.class_transform(self.power_list, self.operation_list, target)
            total_transform = T.Compose(self.test_transformation, new_transform)
            img = total_transform(img)

        elif self.transform :
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

    def update_transform(self, class_transform: type):
        self.class_transform=class_transform

    def update_power_list(self, power_list: list):
        self.power_list = power_list

    def update_operation_list(self, operation_list: list):
        self.operation_list = operation_list
