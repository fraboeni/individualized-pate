from dataclasses import dataclass
from platform import architecture

from per_point_pate.models.pytorch.capc_architecutes.vgg import VGG
from per_point_pate.models.pytorch.capc_architecutes.tiny_resnet import ResNet6
from per_point_pate.models.pytorch.capc_architecutes.small_resnet import ResNet8
from per_point_pate.models.pytorch.capc_architecutes.resnet import ResNet10, ResNet12, ResNet14, ResNet16, ResNet18


@dataclass
class PytorchModelArgs:
    architecture: str
    dataset: str
    num_classes: int
    cuda: bool


ARCHITECTURES = {
    'VGG3': VGG,
    'VGG5': VGG,
    'VGG7': VGG,
    'VGG9': VGG,
    'VGG11': VGG,
    'VGG13': VGG,
    'VGG16': VGG,
    'VGG19': VGG,
    'ResNet6': ResNet6,
    'ResNet8': ResNet8,
    'ResNet10': ResNet10,
    'ResNet12': ResNet12,
    'ResNet14': ResNet14,
    'ResNet16': ResNet16,
    'ResNet18': ResNet18,
}


def PytorchModel(args: PytorchModelArgs):
    if args.architecture in ARCHITECTURES.keys():
        return ARCHITECTURES[args.architecture](name='teacher', args=args)
    else:
        raise Exception(f'Unknown architecture: {args.architecture}')
