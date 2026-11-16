DRIVE = "/content/drive/MyDrive/tcc"

BRUTOS       = f"{DRIVE}/dados_brutos"
PADRONIZADOS = f"{DRIVE}/dados_padronizados"
REDES        = f"{DRIVE}/redes_sociais"
ELA_DIR      = f"{DRIVE}/ela"
MODELOS      = f"{DRIVE}/modelos"
RESULTADOS   = f"{DRIVE}/resultados"
DATASET_CSV  = f"{DRIVE}/dataset.csv"
PARES_CSV    = f"{DRIVE}/pares.csv"
RAISE_CSV    = f"{BRUTOS}/raise/RAISE_1k.csv"

SEED          = 42
CROP          = 256
CROPS_TREINO  = 4
ELA_QUALIDADE = 95
JPEG_CONTROLE = 90
CROP_RAISE    = 1024

GERADORES_TREINO = [
    "dalle2", "dalle3", "midjourney-v5",
    "stable-diffusion-1-3", "stable-diffusion-1-4",
    "stable-diffusion-2", "stable-diffusion-xl",
]
GERADORES_OOD = ["firefly"]

SUBPASTAS = [
    "dados_brutos/raise/nef", "dados_brutos/synthbuster", "dados_brutos/celulares",
    "dados_padronizados/original/real", "dados_padronizados/original/ia",
    "dados_padronizados/original/firefly", "dados_padronizados/jpeg90",
    "redes_sociais/whatsapp", "redes_sociais/instagram", "redes_sociais/x",
    "ela/etapa1", "ela/etapa2_celulares", "ela/etapa3_redes",
    "modelos/baseline_rgb", "modelos/cnn_artigo", "modelos/resnet18",
    "resultados/etapa1", "resultados/etapa2_celulares", "resultados/etapa3_redes",
]
