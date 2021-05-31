import typing

__all__ = ['Logger', 'Dataset', 'DataLoader', 'Optimizer', 'Scheduler', 'Criterion', 'Trainer']

Logger = None
Dataset = None
DataLoader = None
Optimizer = None
Scheduler = None
Criterion = None
Trainer = None


# https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
if typing.TYPE_CHECKING:
    from pyutils.general import Logger
    from torchpack.datasets.dataset import Dataset
    from torch.utils.data import DataLoader
    from torch.optim.optimizer import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler as Scheduler
    from torch.nn.modules.loss import _Loss as Criterion
    from torchpack.train import Trainer
