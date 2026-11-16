import torch
import torch.nn as nn
from torchvision import models


class CNNArtigo(nn.Module):
    """CNN de Martin-Rodriguez et al. (2023), adaptada para entrada 256x256x3."""

    def __init__(self, canais=3, classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(canais, 4, 5, padding="same"), nn.BatchNorm2d(4), nn.ReLU(),
            nn.MaxPool2d(5, stride=4),                                   # 256 -> 63
            nn.Conv2d(4, 8, 5, padding="same"), nn.BatchNorm2d(8), nn.ReLU(),
            nn.MaxPool2d(4, stride=4),                                   # 63 -> 15
            nn.Conv2d(8, 16, 5, padding="same"), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, stride=4),                                   # 15 -> 4
            nn.Conv2d(16, 32, 2, padding="same"), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.classificador = nn.Sequential(nn.Flatten(), nn.Linear(32 * 4 * 4, classes))

    def forward(self, x):
        return self.classificador(self.features(x))


class ResNet18(nn.Module):
    """ResNet-18 pré-treinada no ImageNet, com a última camada trocada para 2 classes."""

    def __init__(self, classes=2):
        super().__init__()
        self.rede = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.rede.fc = nn.Linear(self.rede.fc.in_features, classes)
        self.register_buffer("media", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("desvio", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.rede((x - self.media) / self.desvio)


def criar_modelo(nome):
    if nome == "cnn_artigo":
        return CNNArtigo()
    if nome == "resnet18":
        return ResNet18()
    raise ValueError(f"Modelo desconhecido: {nome}")
