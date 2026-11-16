import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shutil


def fixar_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def salvar_seguro(obj, destino, tentativas=5):
    """Salva no disco do Colab e copia para o Drive, tentando de novo se o Drive falhar."""
    temporario = "/content/_checkpoint_tmp.pt"
    torch.save(obj, temporario)
    for t in range(tentativas):
        try:
            shutil.copy(temporario, destino + ".tmp")
            os.replace(destino + ".tmp", destino)
            return
        except OSError as e:
            print(f"  Falha ao salvar no Drive ({e}); tentando de novo em 10s...")
            time.sleep(10)
    raise RuntimeError(f"Não foi possível salvar {destino} depois de {tentativas} tentativas")


def preparar_lote(x, dispositivo, aumentar=False):
    """uint8 (N, H, W, 3) -> float (N, 3, H, W) entre 0 e 1, com flips aleatórios no treino."""
    x = x.to(dispositivo, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    if aumentar:
        m = torch.rand(x.shape[0], device=dispositivo) < 0.5
        x[m] = x[m].flip(3)
        m = torch.rand(x.shape[0], device=dispositivo) < 0.5
        x[m] = x[m].flip(2)
    return x


@torch.no_grad()
def prever(modelo, X, dispositivo="cuda", tamanho_lote=128):
    """Probabilidade de cada crop ser IA."""
    modelo.eval()
    probs = []
    for i in range(0, len(X), tamanho_lote):
        xb = preparar_lote(X[i:i + tamanho_lote], dispositivo)
        with torch.autocast("cuda", dtype=torch.float16):
            saida = modelo(xb)
        probs.append(torch.softmax(saida.float(), 1)[:, 1].cpu())
    return torch.cat(probs).numpy()


@torch.no_grad()
def avaliar(modelo, X, y, dispositivo, criterio, tamanho_lote=128):
    modelo.eval()
    perda, acertos = 0.0, 0
    for i in range(0, len(X), tamanho_lote):
        xb = preparar_lote(X[i:i + tamanho_lote], dispositivo)
        yb = y[i:i + tamanho_lote].to(dispositivo)
        with torch.autocast("cuda", dtype=torch.float16):
            saida = modelo(xb)
        perda += criterio(saida.float(), yb).item() * len(yb)
        acertos += (saida.argmax(1) == yb).sum().item()
    return perda / len(X), acertos / len(X)


def treinar(modelo, X_tr, y_tr, X_val, y_val, pasta, lr, epocas, paciencia,
            tamanho_lote, seed, dispositivo="cuda"):
    os.makedirs(pasta, exist_ok=True)
    modelo.to(dispositivo)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    escalador = torch.amp.GradScaler("cuda")
    criterio = nn.CrossEntropyLoss()

    estado = {"epoca": 0, "melhor_perda": float("inf"), "sem_melhora": 0,
              "historico": [], "concluido": False}

    ultimo = f"{pasta}/ultimo.pt"
    if os.path.exists(ultimo):
        try:
            ck = torch.load(ultimo, map_location=dispositivo, weights_only=False)
        except Exception as e:
            print(f"Checkpoint corrompido ({e}); recomeçando este treino do zero.")
            os.remove(ultimo)
            ck = None
        if ck is not None:
            modelo.load_state_dict(ck["modelo"])
            otimizador.load_state_dict(ck["otimizador"])
            escalador.load_state_dict(ck["escalador"])
            estado = ck["estado"]
            if estado["concluido"]:
                print("Este treino já foi concluído.")
                return pd.DataFrame(estado["historico"])
            print(f"Retomando a partir da época {estado['epoca'] + 1}")

    n = len(X_tr)
    for epoca in range(estado["epoca"], epocas):
        inicio = time.time()
        modelo.train()
        ordem = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1000 + epoca))

        perda_total, acertos = 0.0, 0
        for i in range(0, n, tamanho_lote):
            idx = ordem[i:i + tamanho_lote]
            xb = preparar_lote(X_tr[idx], dispositivo, aumentar=True)
            yb = y_tr[idx].to(dispositivo)

            otimizador.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                saida = modelo(xb)
                perda = criterio(saida, yb)
            escalador.scale(perda).backward()
            escalador.step(otimizador)
            escalador.update()

            perda_total += perda.item() * len(idx)
            acertos += (saida.argmax(1) == yb).sum().item()

        perda_val, acc_val = avaliar(modelo, X_val, y_val, dispositivo, criterio)
        estado["historico"].append({
            "epoca": epoca + 1,
            "perda_treino": perda_total / n, "acc_treino": acertos / n,
            "perda_val": perda_val, "acc_val": acc_val,
            "segundos": round(time.time() - inicio, 1),
        })
        estado["epoca"] = epoca + 1

        marca = ""
        if perda_val < estado["melhor_perda"]:
            estado["melhor_perda"] = perda_val
            estado["sem_melhora"] = 0
            salvar_seguro(modelo.state_dict(), f"{pasta}/melhor.pt")
            marca = "  <- melhor"
        else:
            estado["sem_melhora"] += 1

        parar_cedo = estado["sem_melhora"] >= paciencia
        estado["concluido"] = parar_cedo or estado["epoca"] >= epocas

        salvar_seguro({"modelo": modelo.state_dict(), "otimizador": otimizador.state_dict(),
                       "escalador": escalador.state_dict(), "estado": estado}, ultimo)
        pd.DataFrame(estado["historico"]).to_csv(f"{pasta}/historico.csv", index=False)

        h = estado["historico"][-1]
        print(f"Época {h['epoca']:2d} | treino: perda {h['perda_treino']:.4f} acc {h['acc_treino']:.3f}"
              f" | val: perda {h['perda_val']:.4f} acc {h['acc_val']:.3f} | {h['segundos']}s{marca}")

        if parar_cedo:
            print(f"Early stopping: {paciencia} épocas sem melhora na validação.")
            break

    return pd.DataFrame(estado["historico"])
