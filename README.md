# 🧠 Multilayer Perceptron (MLP)

Este repositório contém a implementação de um MLP (Multilayer Perceptron), um modelo de rede neural artificial com no mínimo três camadas (camada de entrada, uma ou mais camadas ocultas e uma camada de saída). O projeto foi desenvolvido como parte da disciplina de Redes Neurais Artificiais.

## 🚀 Status do Projeto  
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)

## 🔬 Sobre o Projeto

Uma Multilayer Perceptron (MLP) é uma rede neural profunda composta por múltiplos Perceptrons (neurônios artificiais simples). Ela serviu como precursora para diversas arquiteturas modernas de redes neurais.

## 🎯 Requisitos

### Implementação da MLP

- Desenvolvida manualmente com nn.Module do PyTorch;
- Permite definir a arquitetura da rede: número de entradas, neurônios por camada oculta, quantidade de camadas ocultas e número de saídas;
- Suporte a funções de ativação Tanh ou ReLU nas camadas ocultas.

### Treinamento da MLP

- Treinamento com otimizadores SGD ou Adam;
- Funções de perda adaptadas ao problema: Binary Cross Entropy (binário) ou Cross Entropy (multiclasse);
- Suporte a conjunto de validação (opcional);
- Implementação de Early Stopping (opcional);
- Monitoramento contínuo de acurácia e perda.

### Conjunto de dados

- Foram utilizados os datasets 1, 2 e 4 fornecidos pelo docente:
  - **Dataset 1**: Classificação binária linearmente separável;
  - **Dataset 2**: Classificação binária não-linearmente separável;
  - **Dataset 4**: Problema multiclasse com 50 atributos;
- O modelo foi ajustado de acordo com as características de cada dataset, explorando diferentes combinações de parâmetros para obter os melhores resultados.

## 🛠 Tecnologias Utilizadas

- Python 3.13.0
- PyTorch 2.7.0
- NumPy 2.2.5
- Pandas 2.2.3
- Matplotlib 3.10.0

## 📊 Resultados e Visualizações

O projeto inclui:
- Visualizações dos conjuntos de treino e teste;
- Curvas de perda e acurácia;
- Fronteiras de decisão para avaliação visual do desempenho.

## 🤝 Contribuições

Contribuições são bem-vindas =D

## 📁 Estrutura do Projeto

```plaintext
.
├── data/
│   ├── raw/              # Dados brutos
│   └── processed/        # Dados processados
├── notebooks/            # Notebooks Jupyter
├── src/                  # Código-fonte
│   ├── models/           # Classes de modelos
│   ├── utils/            # Funções auxiliares
│   └── __init__.py       
├── trained_models/       # Modelos treinados
└── README.md