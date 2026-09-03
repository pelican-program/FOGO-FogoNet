import os
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from core.Entities import evaluate_model

class ReportGenerator:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

     
        parts = Path(output_dir).parts
        self.dataset_name = parts[-2] if len(parts) >= 2 else "desconhecido"
        self.model_name = parts[-1] if len(parts) >= 1 else "desconhecido"

        mlflow.set_tracking_uri("sqlite:///C:/mlflow_data/mlflow.db")
        mlflow.set_experiment(f"FOGO-FogoNet")
        self.run = mlflow.start_run(run_name=f"{self.dataset_name}/{self.model_name}")
        mlflow.set_tag("dataset", self.dataset_name)
        mlflow.set_tag("model", self.model_name)

    def summary(self, name, model, val_loader, device, save_model=False):
        txt_header = f'{"-"*15} Avaliação de modelo - {name}{"-"*15}'
        print(txt_header)
        modelNameDir = name.strip().lower().replace(' ', '_')
        model_output_dir = os.path.join(self.output_dir, modelNameDir)
        os.makedirs(model_output_dir, exist_ok=True)

        accuracy, preds, labels = evaluate_model(model, val_loader, device)
        inference_time = self._measure_inference_time(model, val_loader, device)
        model_size = self._get_model_size(model, name)

        print(f"Acurácia: {accuracy:.2f}%")
        print(f"Tempo médio por batch: {inference_time:.4f}s")
        print(f"Tamanho: {model_size:.2f} MB")

        mlflow.log_metric(f"{modelNameDir}_acuracia", accuracy)
        mlflow.log_metric(f"{modelNameDir}_tempo_por_batch", inference_time)
        mlflow.log_metric(f"{modelNameDir}_tamanho_mb", model_size)

        dataset = val_loader.dataset
        classes = dataset.dataset.classes if hasattr(dataset, 'dataset') else dataset.classes
        confusion_matrix_path = self._save_confusion_matrix(name, model_output_dir, labels, preds, classes)

    
        if confusion_matrix_path:
            mlflow.log_artifact(confusion_matrix_path)

        if save_model:
            save_path = os.path.join(model_output_dir, modelNameDir + ".pth")
            torch.save(model.state_dict(), save_path)
            print(f'Modelo salvo em {Path(save_path).absolute()}')
            mlflow.log_artifact(save_path)

        print('-' * len(txt_header))

    def __del__(self):
      
        try:
            mlflow.end_run()
        except:
            pass

    def _measure_inference_time(self, model, data_loader, device):
        model.eval()
        times = []

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                start = time.time()
                _ = model(inputs)[0]
                end = time.time()
                times.append(end - start)

        avg_time = np.mean(times)
        return avg_time

    def _get_model_size(self, model, model_name="Sem Nome"):
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        temp_file = f"temp_{safe_name}.pth"

        try:
            torch.save(model.state_dict(), temp_file)
            size = os.path.getsize(temp_file) / 1024 / 1024  # MB
            os.remove(temp_file)
            return size
        except Exception as e:
            print(f"Erro ao calcular tamanho do modelo '{model_name}': {e}")
            return 0

    def _save_confusion_matrix(self, model_name, output_dir, all_labels, all_preds, class_names):
        # Força a matriz a sempre considerar todas as classes possíveis (0..N-1),
        # mesmo que alguma delas não apareça na amostra de validação dessa rodada.
        labels_idx = list(range(len(class_names)))

        cm = confusion_matrix(all_labels, all_preds, labels=labels_idx)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        save_name = f"matriz_confusao_{model_name.lower().replace(' ', '_')}"
        plt.figure(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f"Matriz de Confusão - {model_name}")
        png_path = os.path.join(output_dir, f"{save_name}.png")
        plt.savefig(png_path)
        plt.close()

        report = classification_report(
            all_labels, all_preds,
            labels=labels_idx,
            target_names=class_names,
            zero_division=0,
        )
        with open(os.path.join(output_dir, f"relatorio_{model_name.lower().replace(' ', '_')}.txt"), "w") as f:
            f.write(f"=== {model_name} ===\n")
            f.write(report)

        np.savetxt(os.path.join(output_dir, f"{save_name}.csv"),
                   cm, delimiter=",", fmt="%d")
        print(f'Matriz de confusão salva em {Path(output_dir).joinpath(save_name + ".png").absolute()}')

        return png_path