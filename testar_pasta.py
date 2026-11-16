"""
Testa as imagens de uma pasta com os modelos treinados no TCC.

Uso:
    python testar_pasta.py --pasta "C:/minhas_fotos"
    python testar_pasta.py --pasta "C:/minhas_fotos" --filtro resnet18
"""
import argparse
import io
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

EXTENSOES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
CROP = 256
N_CROPS = 4


# ---------- ELA (idêntico ao src/ela.py) ----------

def calcular_ela(img, qualidade=95):
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=qualidade, subsampling=0)
    buf.seek(0)
    with Image.open(buf) as rec:
        rec = np.asarray(rec.convert("RGB"), dtype=np.float32)
    return np.abs(np.asarray(img, dtype=np.float32) - rec)


def escalar(ela, fator):
    return np.clip(ela * fator, 0, 255).astype(np.uint8)


def posicoes_grade(altura, largura, tamanho, n):
    lado = int(np.ceil(np.sqrt(n)))
    ys = np.linspace(0, altura - tamanho, lado).astype(int)
    xs = np.linspace(0, largura - tamanho, lado).astype(int)
    return [(int(y), int(x)) for y in ys for x in xs][:n]


# ---------- Modelos (idênticos ao src/modelos.py) ----------

class CNNArtigo(nn.Module):
    def __init__(self, canais=3, classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(canais, 4, 5, padding="same"), nn.BatchNorm2d(4), nn.ReLU(),
            nn.MaxPool2d(5, stride=4),
            nn.Conv2d(4, 8, 5, padding="same"), nn.BatchNorm2d(8), nn.ReLU(),
            nn.MaxPool2d(4, stride=4),
            nn.Conv2d(8, 16, 5, padding="same"), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, stride=4),
            nn.Conv2d(16, 32, 2, padding="same"), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.classificador = nn.Sequential(nn.Flatten(), nn.Linear(32 * 4 * 4, classes))

    def forward(self, x):
        return self.classificador(self.features(x))


class ResNet18(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.rede = models.resnet18(weights=None)   # os pesos vêm do melhor.pt
        self.rede.fc = nn.Linear(self.rede.fc.in_features, classes)
        self.register_buffer("media", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("desvio", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.rede((x - self.media) / self.desvio)


def criar_modelo(nome):
    return {"cnn_artigo": CNNArtigo, "resnet18": ResNet18}[nome]()


# ---------- Processamento ----------

def preparar_crops(caminho, fator, qualidade):
    with Image.open(caminho) as img:
        formato = img.format
        img = img.convert("RGB")
        rgb = np.asarray(img)
        ela = escalar(calcular_ela(img, qualidade), fator)

    altura, largura = rgb.shape[:2]
    if altura < CROP or largura < CROP:
        return None

    posicoes = posicoes_grade(altura, largura, CROP, N_CROPS)
    return {
        "formato": formato, "largura": largura, "altura": altura,
        "rgb": np.stack([rgb[y:y + CROP, x:x + CROP] for y, x in posicoes]),
        "ela": np.stack([ela[y:y + CROP, x:x + CROP] for y, x in posicoes]),
    }


@torch.no_grad()
def prob_ia(modelo, crops):
    x = torch.from_numpy(crops).permute(0, 3, 1, 2).float().div(255)
    return torch.softmax(modelo(x), 1)[:, 1].mean().item()


def main():
    p = argparse.ArgumentParser(description="Testa imagens com os modelos do TCC")
    p.add_argument("--pasta", required=True, help="Pasta com as imagens a testar")
    p.add_argument("--modelos", default=r"H:\Meu Drive\tcc\modelos")
    p.add_argument("--fator", default=r"H:\Meu Drive\tcc\ela\etapa1\fator_ela.json")
    p.add_argument("--saida", default="resultado_teste.csv")
    p.add_argument("--filtro", default="", help="Só modelos cujo nome contém este texto (ex.: resnet18_ela)")
    args = p.parse_args()

    torch.set_num_threads(os.cpu_count())
    info_fator = json.load(open(args.fator))
    fator, qualidade = info_fator["fator"], info_fator["qualidade"]

    # 1. Prepara os crops de todas as imagens
    arquivos = sorted(f for f in os.listdir(args.pasta) if f.lower().endswith(EXTENSOES))
    print(f"{len(arquivos)} imagens encontradas em {args.pasta}")

    crops, resultados = {}, {}
    for arq in arquivos:
        try:
            c = preparar_crops(os.path.join(args.pasta, arq), fator, qualidade)
        except Exception as e:
            print(f"  Pulando {arq}: não foi possível abrir ({e})")
            continue
        if c is None:
            print(f"  Pulando {arq}: menor que {CROP}x{CROP} pixels")
            continue
        crops[arq] = c
        resultados[arq] = {"formato": c["formato"], "largura": c["largura"], "altura": c["altura"]}

    if not crops:
        print("Nenhuma imagem válida.")
        return

    # 2. Roda cada modelo em todas as imagens
    rodadas = sorted(
        d for d in os.listdir(args.modelos)
        if os.path.exists(os.path.join(args.modelos, d, "melhor.pt"))
        and "amostra" not in d and args.filtro in d
    )
    print(f"{len(rodadas)} modelos encontrados\n")

    for i, nome in enumerate(rodadas, 1):
        pasta = os.path.join(args.modelos, nome)
        cfg = json.load(open(os.path.join(pasta, "config.json")))
        modelo = criar_modelo(cfg["modelo"])
        modelo.load_state_dict(torch.load(os.path.join(pasta, "melhor.pt"), map_location="cpu", weights_only=True))
        modelo.eval()
        print(f"[{i}/{len(rodadas)}] {nome}")
        for arq, c in crops.items():
            resultados[arq][nome] = prob_ia(modelo, c[cfg["entrada"]])

    # 3. Salva o resultado completo e um resumo
    df = pd.DataFrame.from_dict(resultados, orient="index")
    df.index.name = "arquivo"
    df.to_csv(args.saida)

    configs = sorted({n.rsplit("_seed", 1)[0] for n in rodadas})
    resumo = pd.DataFrame({
        c: df[[n for n in rodadas if n.startswith(c + "_seed")]].mean(axis=1) for c in configs
    })
    votos = (df[rodadas] >= 0.5).sum(axis=1)
    resumo["votos_ia"] = votos.astype(str) + "/" + str(len(rodadas))
    resumo["maioria"] = np.where(votos > len(rodadas) / 2, "IA", "real")
    resumo.to_csv(args.saida.replace(".csv", "_resumo.csv"))

    pd.set_option("display.width", 250)
    print("\nProbabilidade de ser IA (média das seeds de cada configuração):")
    print(resumo.round(2).to_string())
    print(f"\nResultado completo: {args.saida}")
    print(f"Resumo: {args.saida.replace('.csv', '_resumo.csv')}")


if __name__ == "__main__":
    main()