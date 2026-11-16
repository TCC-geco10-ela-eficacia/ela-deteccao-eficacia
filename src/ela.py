import io
import numpy as np
from PIL import Image


def calcular_ela(img, qualidade=95):
    """Diferença absoluta entre a imagem e ela mesma recomprimida em JPEG."""
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=qualidade, subsampling=0)
    buf.seek(0)
    with Image.open(buf) as rec:
        rec = np.asarray(rec.convert("RGB"), dtype=np.float32)
    return np.abs(np.asarray(img, dtype=np.float32) - rec)


def escalar(ela, fator):
    """Converte o ELA para uint8 com um fator fixo, igual para todas as imagens."""
    return np.clip(ela * fator, 0, 255).astype(np.uint8)


def posicoes_grade(altura, largura, tamanho, n):
    """Posições fixas, espalhadas pela imagem (usadas na validação e no teste)."""
    lado = int(np.ceil(np.sqrt(n)))
    ys = np.linspace(0, altura - tamanho, lado).astype(int)
    xs = np.linspace(0, largura - tamanho, lado).astype(int)
    return [(int(y), int(x)) for y in ys for x in xs][:n]


def posicoes_aleatorias(altura, largura, tamanho, n, rng):
    """Posições sorteadas (usadas no treino)."""
    return [
        (int(rng.integers(0, altura - tamanho + 1)), int(rng.integers(0, largura - tamanho + 1)))
        for _ in range(n)
    ]
