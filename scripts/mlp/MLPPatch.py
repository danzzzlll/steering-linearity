import torch
from torch import nn
from contextlib import contextmanager
from typing import List, Dict

class _DynamicMLPPatch:
    """Внутренний класс-контекстный менеджер, который ставит и снимает хуки."""
    def __init__(self,
                 model: nn.Module,
                 mlp_model: nn.Module,
                 layer_ids: List[int],
                 multiplier: float = 1.0,
                 layer_name_template: str = "model.layers.{}"):
        self.model = model
        self.mlp_model = mlp_model
        self.multiplier = multiplier
        # Теперь мы генерируем имена слоев сами
        self.target_layers = [layer_name_template.format(i) for i in layer_ids]
        self.handles = []

        # Проверка, что слои существуют в модели
        all_module_names = {name for name, _ in model.named_modules()}
        for layer_name in self.target_layers:
            if layer_name not in all_module_names:
                raise ValueError(
                    f"Слой '{layer_name}' не найден в модели. "
                    f"Возможно, вам нужно изменить 'layer_name_template'. "
                    f"Доступные модули: {[n for n in all_module_names if 'layer' in n or 'block' in n or 'h' in n][:10]}..."
                )

    def __enter__(self):
        hook_function = self._create_hook()
        # Получаем словарь всех модулей для быстрого доступа
        modules = dict(self.model.named_modules())
        for layer_name in self.target_layers:
            module = modules[layer_name]
            handle = module.register_forward_hook(hook_function)
            self.handles.append(handle)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _create_hook(self):
        """Хук берёт активации, прогоняет их через SteerMLP и вносит скорректированные."""
        def hook(module, args, output):
            # output может быть tuple, но для MLP-слоёв у HF это обычно тензор
            act = output if not isinstance(output, tuple) else output[0]
            dtype = act.dtype
            with torch.no_grad():
                # mlp_model возвращает identity + gate * delta
                corrected = self.mlp_model(act.to(torch.float32))
                # масштабируем силу коррекции
                new_act = act.to(torch.float32) + self.multiplier * (corrected - act.to(torch.float32))
            new_act = new_act.to(dtype)
            # если output — tuple, возвращаем tuple
            if isinstance(output, tuple):
                return (new_act, *output[1:])
            else:
                return new_act
        return hook


class DynamicMLPSteering:
    """Основной класс для управления динамическим стирингом через MLP."""
    def __init__(self, mlp_model: nn.Module, device: str = "cpu"):
        self.mlp_model = mlp_model.to(device)
        self.mlp_model.eval()

    @contextmanager
    def apply(self,
              model: nn.Module,
              layers: List[int],
              multiplier: float = 1.0,
              layer_name_template: str = "model.layers.{}"):
        """
        Применяет динамический стиринг к модели.

        :param model: Языковая модель для модификации.
        :param layers: Список ID слоев для применения стиринга.
        :param multiplier: Сила стиринга.
        :param layer_name_template: Шаблон имени слоя. Зависит от архитектуры модели.
        """
        patch = _DynamicMLPPatch(model, self.mlp_model, layers, multiplier, layer_name_template)
        with patch:
            yield