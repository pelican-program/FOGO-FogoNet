import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset as TorchDataset, Subset

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Remove sequências de escape ANSI (ex: '\x1b[20;1R') que alguns terminais
# injetam no stdin junto com a resposta do usuário.
_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')


def _clean_input(prompt):
    raw = input(prompt)
    return _ANSI_ESCAPE_RE.sub('', raw).strip()


def _clean_int_input(prompt):
    cleaned = _clean_input(prompt)
    match = re.search(r'-?\d+', cleaned)
    if not match:
        raise ValueError(f"Entrada inválida: '{cleaned}'")
    return int(match.group())


def _clean_float_input(prompt):
    cleaned = _clean_input(prompt)
    match = re.search(r'-?\d+(\.\d+)?', cleaned)
    if not match:
        raise ValueError(f"Entrada inválida: '{cleaned}'")
    return float(match.group())


class FogoImageDataset(TorchDataset):
    """
    Dataset que lê imagens soltas (sem subpastas) de um diretório e define
    a classe de cada imagem pelo nome do arquivo:
      - se o nome contém 'no_fire' -> classe 'no_fire'
      - caso contrário -> classe 'fire'
    """

    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform

        # Mantém o mesmo formato que o ImageFolder usava (para compatibilidade com o main.py e ReportGenerator.py)
        self.class_to_idx = {'fire': 0, 'no_fire': 1}
        self.classes = sorted(self.class_to_idx, key=self.class_to_idx.get)

        self.samples = []
        for entry in sorted(os.listdir(self.folder_path)):
            full_path = self.folder_path / entry
            if full_path.is_file() and entry.lower().endswith(IMG_EXTENSIONS):
                label_name = 'no_fire' if 'no_fire' in entry.lower() else 'fire'
                self.samples.append((full_path, self.class_to_idx[label_name]))

        if len(self.samples) == 0:
            raise RuntimeError(f'Nenhuma imagem encontrada em {self.folder_path}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class Dataset:
    def __init__(self, name, root_path, train_split=0.8):
        self.name = name
        self.root_path = Path(root_path)
        self.train_split = train_split

        self.train_tfms = None
        self.test_tfms = None

    def define_train_transform(self, transform_model):
        self.train_tfms = transform_model

    def define_test_transform(self, transform_model):
        self.test_tfms = transform_model

    def get(self, transform_model=None):
        if transform_model == "inception-format":
            self.train_tfms = transforms.Compose([
                transforms.RandomResizedCrop(299, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

            self.test_tfms = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        elif transform_model == "googlenet-format":
            pass

        # Dois datasets separados (um por transform) para evitar compartilhar o mesmo objeto
        full_dataset_train = FogoImageDataset(self.root_path / self.name, transform=self.train_tfms)
        full_dataset_val = FogoImageDataset(self.root_path / self.name, transform=self.test_tfms)

        total = len(full_dataset_train)
        train_size = int(total * self.train_split)

        indices = np.random.permutation(total)
        train_idx, val_idx = indices[:train_size], indices[train_size:]

        train_data = Subset(full_dataset_train, train_idx)
        val_data = Subset(full_dataset_val, val_idx)

        print(f"Total de imagens: {total} | Treino: {len(train_idx)} | Teste: {len(val_idx)}")

        return train_data, val_data


def _get_datasets(datasets_root_path):
    result = []
    for folder in os.listdir(datasets_root_path):
        full_path = os.path.join(datasets_root_path, folder)
        if os.path.isdir(full_path):
            has_images = any(
                f.lower().endswith(IMG_EXTENSIONS)
                for f in os.listdir(full_path)
                if os.path.isfile(os.path.join(full_path, f))
            )
            if has_images:
                result.append(Dataset(folder, datasets_root_path))
    return result


def request_dataset(datasets_path):
    available_datasets = _get_datasets(datasets_path)
    if len(available_datasets) == 0:
        print('Nenhum dataset encontrado!')
        return None

    print('\nDatasets disponiveis:')
    for i, dataset in enumerate(available_datasets):
        print(f'{i + 1}. {dataset.name}')

    datasetOpt = _clean_int_input('Escolha o seu dataset: ')
    while datasetOpt not in range(1, len(available_datasets) + 1):
        print('Opção inválida.')
        datasetOpt = _clean_int_input('Escolha o seu dataset: ')

    selected = available_datasets[datasetOpt - 1]

    train_pct = _clean_float_input('Percentual para treino (ex: 0.8 para 80%): ')
    while not (0 < train_pct < 1):
        print('Valor inválido. Digite um número entre 0 e 1 (ex: 0.8).')
        train_pct = _clean_float_input('Percentual para treino (ex: 0.8 para 80%): ')

    selected.train_split = train_pct
    return selected