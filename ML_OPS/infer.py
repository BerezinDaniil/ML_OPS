import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
from dvc.api import DVCFileSystem
from omegaconf import DictConfig
from torchmetrics import F1Score


@hydra.main(version_base=None, config_path="../configs", config_name="test")
def infer(cfg: DictConfig):

    fs = DVCFileSystem()
    fs.get_file(cfg.data.name, cfg.data.name)

    test = pd.read_csv(cfg.data.name)
    X_test = np.array(test.drop(columns=cfg.data.target), dtype=np.float32)
    y_test = test[cfg.data.target].values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    model = mlflow.pytorch.load_model(f"models:/{cfg.model.name}/latest")
    model = model.to(device)
    model.eval()
    outputs = model(torch.from_numpy(X_test).float().to(device))
    f1 = F1Score(task=cfg.model.f1_task, num_classes=cfg.model.output_dim)
    _, predict = torch.max(outputs.data, 1)
    labels = torch.from_numpy(y_test).float().to(device)
    test_acc = torch.sum(labels == predict).item() / (len(predict) * 1.0)
    test_f1 = f1(predict, labels).item()
    print(f"test_acc: {test_acc:.4f}, test_f1: {test_f1:.4f}")


if __name__ == "__main__":
    infer()
