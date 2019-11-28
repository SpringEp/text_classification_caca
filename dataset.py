from pathlib import Path
from collections import namedtuple

# Dataset root directory
_DATASET_ROOT = Path('./data')

Dataset = namedtuple('Dataset', ['name', 'root', 'neg', 'pos'])

data = Dataset(
    'data',
    _DATASET_ROOT / 'txt_sentoken',
    _DATASET_ROOT / 'txt_sentoken/neg',
    _DATASET_ROOT / 'txt_sentoken/pos'
)

### Current dataset in use. (change this name to change the dataset)
DATASET = data


if __name__ == '__main__':
    print(DATASET.name, DATASET.root, DATASET.neg, DATASET.pos)