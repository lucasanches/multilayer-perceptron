# src/utils/model_io.py

import os
import json
from datetime import datetime
import torch


def save_model(model, model_name, config=None, directory="trained_models"):
    os.makedirs(directory, exist_ok=True)
    os.makedirs(os.path.join(directory, "configs"), exist_ok=True)

    # Nomeando o arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_name = f"{model_name}_{timestamp}"

    # Caminho completo para salvar modelo
    model_path = os.path.join(directory, f"{full_name}.pt")
    config_path = os.path.join(directory, "configs", f"{full_name}_config.json")

    # Salva modelo
    torch.save(model.state_dict(), model_path)
    print(f"Modelo salvo em: {model_path}")

    # Salva configuração se fornecida
    if config:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuração salva em: {config_path}")

    return model_path, config_path


def load_model(model_class, model_path, config_path):
    with open(config_path) as f:
        config = json.load(f)

    model = model_class(**config)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model