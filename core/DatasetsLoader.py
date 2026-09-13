import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset as TorchDataset, Subset

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

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
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform

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
    def __init__(self, name, root_path, fire_split=0.8, no_fire_split=0.8):
        self.name = name
        self.root_path = Path(root_path)
        self.fire_split = fire_split
        self.no_fire_split = no_fire_split

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

        full_dataset_train = FogoImageDataset(self.root_path / self.name, transform=self.train_tfms)
        full_dataset_val = FogoImageDataset(self.root_path / self.name, transform=self.test_tfms)

        fire_idx = [i for i, (_, label) in enumerate(full_dataset_train.samples) if label == 0]
        no_fire_idx = [i for i, (_, label) in enumerate(full_dataset_train.samples) if label == 1]

        fire_idx = np.random.permutation(fire_idx).astype(int)
        no_fire_idx = np.random.permutation(no_fire_idx).astype(int)

        fire_train_size = int(len(fire_idx) * self.fire_split)
        no_fire_train_size = int(len(no_fire_idx) * self.no_fire_split)

        fire_train_idx = fire_idx[:fire_train_size]
        fire_val_idx = fire_idx[fire_train_size:]

        no_fire_train_idx = no_fire_idx[:no_fire_train_size]
        no_fire_val_idx = no_fire_idx[no_fire_train_size:]

        train_idx = np.concatenate([fire_train_idx, no_fire_train_idx])
        val_idx = np.concatenate([fire_val_idx, no_fire_val_idx])

        train_data = Subset(full_dataset_train, train_idx)
        val_data = Subset(full_dataset_val, val_idx)

        print(f"\nFire    — Total: {len(fire_idx)} | Treino: {fire_train_size} | Teste: {len(fire_val_idx)}")
        print(f"No Fire — Total: {len(no_fire_idx)} | Treino: {no_fire_train_size} | Teste: {len(no_fire_val_idx)}")
        print(f"Total   — Treino: {len(train_idx)} | Teste: {len(val_idx)}")

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

    print('\nDefina o percentual de treino para cada classe:')

    fire_pct = _clean_float_input('Percentual de treino para fire: ')
    while not (0 < fire_pct < 1):
        print('Valor inválido. Digite um número entre 0 e 1.')
        fire_pct = _clean_float_input('Percentual de treino para fire: ')

    no_fire_pct = _clean_float_input('Percentual de treino para no_fire: ')
    while not (0 < no_fire_pct < 1):
        print('Valor inválido. Digite um número entre 0 e 1 (ex: 0.8).')
        no_fire_pct = _clean_float_input('Percentual de treino para no_fire: ')

    selected.fire_split = fire_pct
    selected.no_fire_split = no_fire_pct
    return selected