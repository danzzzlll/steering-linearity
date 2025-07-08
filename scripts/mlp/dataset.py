"""
Dataset building utilities for steering MLP experiments.

This module defines:
- `SteeringDataset` – a torch `Dataset` that wraps (example, layer) pairs
  together with their positive / negative activations.
- `create_train_val_test_loaders` – helper that returns pytorch `DataLoader`s
  for the requested split.
- `print_split_info` – pretty–prints dataset statistics.

The implementation has been lifted directly from the original Jupyter
notebook and lightly refactored into a standalone, importable script.
"""


from torch.utils.data import Dataset, DataLoader
import numpy as np

class SteeringDataset(Dataset):
    def __init__(self, data, layer_indices=-1, target_type='pos',
                 split='train', split_ratios=(0.8, 0.2),
                 split_strategy='by_example', random_seed=42):
        """
        Args:
            data: список из 1000 примеров с 'pos' и 'neg' активациями
            layer_indices: -1 для всех слоев, int для одного, list для нескольких
            target_type: 'pos', 'diff'
            split: 'train', 'val'
            split_ratios: (train_ratio, val_ratio, test_ratio)
            split_strategy: 'sequential', 'by_example'
            random_seed: для воспроизводимости
        """
        self.data = data    
        self.target_type = target_type
        self.split = split
        self.split_ratios = split_ratios
        self.split_strategy = split_strategy
        self.random_seed = random_seed

        if layer_indices == -1:
            self.layers = list(range(32))
        elif isinstance(layer_indices, int):
            self.layers = [layer_indices]
        else:
            self.layers = layer_indices

        self._create_all_indices()

        self._split_data()

    def _create_all_indices(self):
        """Создает все возможные индексы (example_idx, layer_idx)"""
        self.all_indices = []
        for example_idx in range(len(self.data)):
            for layer_idx in self.layers:
                self.all_indices.append((example_idx, layer_idx))

    def _split_data(self):
        """Разделяет данные на train/val"""

        np.random.seed(self.random_seed)

        if self.split_strategy == 'sequential':
            self._sequential_split()
        else:
            self._example_based_split()

    def _sequential_split(self):
        """Последовательное разделение по примерам"""
        n_examples = len(self.data)
        n_train = int(n_examples * self.split_ratios[0])

        if self.split == 'train':
            example_indices = list(range(n_train))
        else:
            example_indices = list(range(n_train, n_examples))

        self.indices = []
        for example_idx in example_indices:
            for layer_idx in self.layers:
                self.indices.append((example_idx, layer_idx))

    def _example_based_split(self):
        """все слои одного примера в одном split"""
        example_indices = list(range(len(self.data)))
        np.random.shuffle(example_indices)

        n_examples = len(example_indices)
        n_train = int(n_examples * self.split_ratios[0])
        n_val = int(n_examples * self.split_ratios[1])

        if self.split == 'train':
            selected_examples = example_indices[:n_train]
        else:
            selected_examples = example_indices[n_train:n_train + n_val]

        self.indices = []
        for example_idx in selected_examples:
            for layer_idx in self.layers:
                self.indices.append((example_idx, layer_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        example_idx, layer_idx = self.indices[idx]
        example = self.data[example_idx]

        neg_activation = example['neg'][layer_idx].squeeze()
        pos_activation = example['pos'][layer_idx].squeeze()

        if self.target_type == 'pos':
            return neg_activation, pos_activation
        else:
            return neg_activation, pos_activation - neg_activation

    def get_split_info(self):
        return {
            'split': self.split,
            'n_samples': len(self.indices),
            'n_examples_used': len(set(idx[0] for idx in self.indices)),
            'n_layers_used': len(set(idx[1] for idx in self.indices)),
            'layers': self.layers,
            'split_strategy': self.split_strategy
        }
    

def create_train_val_test_loaders(data, layer_indices=[2, 4], batch_size=32,
                                 target_type='pos', split_ratios=(0.8, 0.2),
                                 split_strategy='by_example', random_seed=42):
    """
    Создает train, val, test DataLoader'ы

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Создаем датасеты
    train_dataset = SteeringDataset(
        data, layer_indices, target_type, 'train',
        split_ratios, split_strategy, random_seed
    )
    val_dataset = SteeringDataset(
        data, layer_indices, target_type, 'val',
        split_ratios, split_strategy, random_seed
    )

    # Создаем DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def print_split_info(train_loader, val_loader):
    """Выводит информацию о разделении данных"""
    print("=== Информация о разделении данных ===")

    for name, loader in [("Train", train_loader), ("Val", val_loader)]:
        info = loader.dataset.get_split_info()
        print(f"\n{name}:")
        print(f"  Семплов: {info['n_samples']}")
        print(f"  Примеров: {info['n_examples_used']}")
        print(f"  Слоев: {info['n_layers_used']}")
        print(f"  Стратегия: {info['split_strategy']}")


# train_loader, val_loader = create_train_val_test_loaders(
#     data=super_nervous,
#     layer_indices=[14],
#     batch_size=64,
#     target_type='pos',
#     split_ratios=(0.8, 0.2),
#     split_strategy='by_example',
#     random_seed=42
# )

# print_split_info(train_loader, val_loader)