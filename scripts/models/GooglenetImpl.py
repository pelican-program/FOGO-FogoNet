import torch
from torch import nn
from torchvision import models
import torch.quantization as quantization

from core.Entities import evaluate_model


def display_name():
    return "Googlenet"


def init():
    model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
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
            result = model(inputs)
            if isinstance(result, tuple):
                outputs, aux_outputs = result
                loss = criterion(outputs, labels) + 0.4 * criterion(aux_outputs, labels)
            else:
                outputs = result.logits if hasattr(result, 'logits') else result
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
    try:
        model_static = models.googlenet(pretrained=True)
        model_static.fc = torch.nn.Linear(model_static.fc.in_features, 2)
        model_static.load_state_dict(model_cpu.state_dict())

        model_static.eval()

        backends = ['fbgemm', 'qnnpack']
        model_static_quantized = None

        for backend in backends:
            try:
                print(f"Tentando quantização com backend: {backend}")
                model_static.qconfig = quantization.get_default_qconfig(backend)

                model_static_prepared = quantization.prepare(model_static, inplace=False)

                print("Calibrando modelo com dados de validação...")
                with torch.no_grad():
                    for i, (inputs, _) in enumerate(val_loader):
                        if i >= 10:
                            break
                        model_static_prepared(inputs)
                        if i % 5 == 0:
                            print(f"Calibração: {i + 1}/10 batches processados")

                model_static_quantized = quantization.convert(model_static_prepared, inplace=False)
                print(f"Quantização estática aplicada com sucesso usando backend: {backend}!")

                report_generator.summary("Quantizado Estático", model_static_quantized, val_loader, torch.device("cpu"))

                return model_static_quantized

            except Exception as e:
                print(f"Erro com backend {backend}: {e}")
                continue
    except Exception as e:
        print(f"ERRO na quantização estática: {e}")
        print("Continuando apenas com quantização dinâmica...")

    return None
