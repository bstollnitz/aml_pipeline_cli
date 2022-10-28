"""Tests the model."""

import argparse
import logging

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from common import DATA_DIR, MODEL_DIR
from utils_train_nn import evaluate


def load_test_data(data_dir: str, batch_size: int) -> DataLoader[torch.Tensor]:
    """
    Returns a DataLoader object that wraps test data.
    """
    test_data = datasets.FashionMNIST(data_dir,
                                      train=False,
                                      download=True,
                                      transform=ToTensor())
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return test_loader


def test(data_dir: str, model_dir: str, device: str) -> None:
    """
    Tests the model on test data.
    """
    batch_size = 64
    loss_fn = nn.CrossEntropyLoss()
    test_dataloader = load_test_data(data_dir, batch_size)
    model = mlflow.pytorch.load_model(model_uri=model_dir)

    (test_loss, test_accuracy) = evaluate(device, test_dataloader, model,
                                          loss_fn)

    mlflow.log_param("test_loss", test_loss)
    mlflow.log_param("test_accuracy", test_accuracy)
    logging.info("Test loss: %f", test_loss)
    logging.info("Test accuracy: %f", test_accuracy)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--model_dir", dest="model_dir", default=MODEL_DIR)
    args = parser.parse_args()
    logging.info("input parameters: %s", vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test(**vars(args), device=device)


if __name__ == "__main__":
    main()
