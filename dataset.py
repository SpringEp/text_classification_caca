from pathlib import Path
from collections import namedtuple

# Dataset root directory
_DATASET_ROOT = Path('./data')

Dataset = namedtuple('Dataset', ['name', 'root', 'neg', 'pos'])

# # Source codes and bug repositories
# aspectj = Dataset(
#     'aspectj',
#     _DATASET_ROOT / 'AspectJ',
#     _DATASET_ROOT / 'AspectJ/AspectJ-1.5',
#     _DATASET_ROOT / 'AspectJ/AspectJBugRepository.xml'
# )
#
# swt = Dataset(
#     'swt',
#     _DATASET_ROOT / 'SWT',
#     _DATASET_ROOT / 'SWT/SWT-3.1',
#     _DATASET_ROOT / 'SWT/SWTBugRepository.xml'
# )
#
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