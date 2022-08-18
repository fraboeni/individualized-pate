from typing import Tuple
from torchvision import transforms


def get_transform(dataset: str, augmentation=False):
    shapes = {
        'mnist': (1, 28, 28),
        'fashion_mnist': (1, 28, 28),
        'cifar10': (3, 32, 32),
        'svhn': (3, 32, 32),
    }

    distributions = {
        'mnist': {
            'mean': (0.1307, ),
            'std': (0.3081, ),
        },
        'fashion_mnist': {
            'mean': (0.2859, ),
            'std': (0.3530, ),
        },
        'cifar10': {
            'mean': (0.49139969, 0.48215842, 0.44653093),
            'std': (0.24665252, 0.24289226, 0.26159238),
        },
        'svhn': {
            'mean': (0.4376821, 0.4437697, 0.47280442),
            'std': (0.19803012, 0.20101562, 0.19703614),
        }
    }

    shape = shapes[dataset]
    n_channels, height, width = shape

    # basic transformation to normalized tensor
    transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean=distributions[dataset]['mean'],
                             std=distributions[dataset]['std']),
    ]

    # addign data augmentation for training
    if augmentation:
        transform.extend([
            transforms.RandomRotation(degrees=10, ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                size=(height, width),
                scale=(0.9, 1.0),
                ratio=(1.0, 1.0),
            ),
        ])
    return transforms.Compose(transform)
