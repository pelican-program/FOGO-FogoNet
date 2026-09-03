import sys
from pathlib import Path

import torch
from torchvision import models, transforms, datasets
from torchvision.models import Inception_V3_Weights
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import time
import numpy as np

from core.Entities import evaluate_model, get_model_options, request_model
from core.DatasetsLoader import request_dataset
from core.ModelQuantizer import ModelQuantizer
from core.ReportGenerator import ReportGenerator

DATASETS_ROOT_PATH = os.path.join('.', "datasets")
if __name__ == "__main__":
    print('Iniciando..')

    dataset = request_dataset(DATASETS_ROOT_PATH)
    if not dataset:
        print("Dê uma olhada no arquivo 'SETUP' na pasta", Path(DATASETS_ROOT_PATH).absolute())
        sys.exit(1)

    train_data, val_data = dataset.get(transform_model='inception-format')
    print('Classes do dataset:', train_data.dataset.class_to_idx)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=False)

    model, model_name, criterion, optimizer, train_fn = request_model()

    should_quantize_dynamically, should_quantize_statically = get_model_options()

    reportGenerator = ReportGenerator(output_dir=f"results/{dataset.name}/{model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    model = model.to(device)

    max_epochs = int(input('Quantas epocas deseja treinar?'))
    while max_epochs <= 0:
        print('Número de epocas inválido, tente novamente.')
        max_epochs = int(input('Quantas epocas deseja treinar?'))

    if callable(train_fn):
        model = train_fn(max_epochs=max_epochs,
                         device=device,
                         model=model,
                         optimizer=optimizer,
                         criterion=criterion,
                         train_loader=train_loader,
                         val_loader=val_loader
                         )
    else:
        print('Função de treino não encontrada na implementação do modelo. Nenhum treino foi realizado.')
        sys.exit(1)
    print('Gerando os resultados...')
    reportGenerator.summary("Original", model, val_loader, device, save_model=True)

    # Quantização
    quantizer = ModelQuantizer([model_name, model], val_loader, reportGenerator)
    if should_quantize_dynamically:
        quantizer.dynamic()
    if should_quantize_statically:
        quantizer.static()