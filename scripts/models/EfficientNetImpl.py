import torch
from torch import nn
from torchvision import models

from core.Entities import evaluate_model


def display_name():
    return "EfficientNet"


def init():
    model = models.efficientnet_b0(pretrained=False)
    state_dict = torch.hub.load_state_dict_from_url(
        "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
        check_hash=False
    )
    model.load_state_dict(state_dict)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    return model, criterion, optimizer


def train(max_epochs, device, model, optimizer, criterion, train_loader, val_loader):
    print('-' * 50)
    print('Treinando modelo...')
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        epoch_summary = f"Epoch {epoch + 1}/{max_epochs} - Loss: {avg_loss:.4f},"
        accuracy, all_preds, all_labels = evaluate_model(model, val_loader, device)
        epoch_summary += f" Accuracy: {accuracy:.2f}%"
        print(epoch_summary)
    print('-' * 50)
    return model


def static_quantize(model_cpu, val_loader, report_generator):
    print("EfficientNet usa ativacao SiLU (Swish), incompativel com quantizacao estatica nesta versao do PyTorch.")
    print("Pulando quantizacao estatica.")
    return None