import os
from pathlib import Path

from torchvision import transforms, datasets
from torch.utils.data import random_split


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

        full_dataset = datasets.ImageFolder(
            self.root_path / self.name,
            transform=self.train_tfms
        )


        total = len(full_dataset)
        train_size = int(total * self.train_split)
        test_size = total - train_size

        train_data, val_data = random_split(full_dataset, [train_size, test_size])

        val_data.dataset.transform = self.test_tfms

        print(f"Total de imagens: {total} | Treino: {train_size} | Teste: {test_size}")

        return train_data, val_data


def _get_datasets(datasets_root_path):
    result = []
    for folder in os.listdir(datasets_root_path):
        full_path = os.path.join(datasets_root_path, folder)
        if os.path.isdir(full_path):
            subfolders = [f for f in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, f))]
            if subfolders:
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

    datasetOpt = int(input('Escolha o seu dataset: '))
    while datasetOpt not in range(1, len(available_datasets) + 1):
        print('Opção inválida.')
        datasetOpt = int(input('Escolha o seu dataset: '))

    selected = available_datasets[datasetOpt - 1]

    train_pct = float(input('Percentual para treino (ex: 0.8 para 80%): '))
    while not (0 < train_pct < 1):
        print('Valor inválido. Digite um número entre 0 e 1 (ex: 0.8).')
        train_pct = float(input('Percentual para treino (ex: 0.8 para 80%): '))

    selected.train_split = train_pct
    return selected