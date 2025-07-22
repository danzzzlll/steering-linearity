from pathlib import Path
from typing import List, Union
import numpy as np
from tqdm import tqdm

class ActivationDataset:
    """
    • В папке лежат несколько .npy-файлов, каждый shape = (N, 32, 4096)
    • При инициализации мы **один раз** загружаем / мем-мапим их
      и склеиваем по оси примеров  ➜ (total_N, 32, 4096)
    • Обращение по индексу возвращает нужный слой:  ds[3] → (total_N, 4096)
    """

    def __init__(self, folder: Union[str, Path], *, mmap: bool = True):
        self.folder = Path(folder)
        files: List[Path] = sorted(self.folder.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files found in {self.folder}")

        arrs = [
            np.load(f, mmap_mode="r" if mmap else None)
            for f in tqdm(files, desc='load files')
        ]

        # for a in arrs:
        #     if a.ndim != 3 or a.shape[1:] != (32, 4096):
        #         raise ValueError(f"{a.shape=} — ожидаю (N, 32, 4096)")

        self.data = np.concatenate(arrs, axis=0)

        self.n_samples, self.n_layers, self.dim = self.data.shape
        self.layers = list(range(self.n_layers))
        print(
            f"Loaded {self.n_samples} samples  ·  "
            f"{self.n_layers} layers  ·  {self.dim}-D vectors"
        )

    def __getitem__(self, layer_idx: Union[int, slice]):
        return self.data[:10000, layer_idx, :]

    def layer_shape(self):
        """Форма конкретного слоя, не вытаскивая сам массив."""
        return self.n_samples, self.dim