# Classe MLP

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

class MLP(nn.Module):

    """
    Classe para definição de uma Rede Neural Perceptron Multicamadas (MLP).

    A rede suporta múltiplas camadas ocultas, ativação ReLU ou Tanh, e é adequada tanto para 
    classificação binária quanto multiclasse. Inclui mecanismos de early stopping, dropout, 
    e regularização L2 (weight decay).

    Args:
        n_epochs (int): Número de épocas para o treinamento.
        n_features (int): Número de features (entradas) no dataset.
        n_hidden (List[int]): Lista com o número de neurônios em cada camada oculta.
        n_classes (int): Número de classes de saída. Use 1 para classificação binária.
        activation (str): Função de ativação ("relu" ou "tanh").
        learning_rate (float): Taxa de aprendizado para o otimizador.
        patience (Optional[int]): Número de épocas sem melhoria antes do early stopping. Default é None.
        weight_decay (Optional[float]): Fator de regularização L2. Default é None.
        dropout (Optional[float]): Taxa de dropout. Default é None.
        seed (Optional[int]): Semente para reprodutibilidade. Default é None.
    """

    def __init__(
        self,
        n_epochs: int,
        n_features: int,
        n_hidden: List[int],
        n_classes: int,
        activation: str,
        learning_rate: float,
        patience: Optional[int] = None,
        weight_decay: Optional[float] = None,
        dropout: Optional[float] = None,
        seed: Optional[int] = None
        ):

        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.sigmoid = nn.Sigmoid()

        self.n_epochs = n_epochs
        self.lr = learning_rate
        self.n_classes = n_classes
        self.patience = patience
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.best_loss = float('inf')
        self.no_improvement_count = 0
        self.stop_training = False

        # Camada de entrada
        self.input_layer = nn.Linear(n_features, n_hidden[0])

        # Camadas escondidas
        self.hidden_layers = nn.ModuleList()

        for layer in range(len(n_hidden) - 1):
            self.hidden_layers.append(nn.Linear(n_hidden[layer], n_hidden[layer + 1]))

        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout is not None else nn.Identity()

        # Camada de saída
        self.output_layer = nn.Linear(n_hidden[-1], n_classes)

        # Inicialização dos pesos
        # for layer in [self.input_layer, *self.hidden_layers, self.output_layer]:
        #     nn.init.xavier_uniform_(layer.weight)
        #     nn.init.zeros_(layer.bias)
        
        # Função de ativação das camadas escondidas
        self.activation_name = activation

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("Função de ativação não reconhecida. Escolha entre 'relu' ou 'tanh'.")
        

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:           # definindo como os dados passam pela rede neural

        """
        Define o caminho (forward pass) da rede.

        Args:
            x (torch.Tensor): Tensor de entrada com shape (batch_size, n_features).

        Returns:
            torch.Tensor: Saída da rede com shape (batch_size, n_classes).
        """

        x = self.activation(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            x = self.dropout_layer(x) if self.training else x       # Aplica o dropout caso necessário

        x = self.output_layer(x)

        return x
    


    def check_early_stop(self, val_loss: float) -> None:

        """
        Atualiza o estado do early stopping com base na perda de validação.

        Args:
            val_loss (float): Valor atual da perda no conjunto de validação.
        """

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                print("Parando treinamento por early stopping.")


    
    def compute_accuracy(
        self,
        output: torch.Tensor, 
        y_test: torch.Tensor,
        n_classes: int
        ) -> float:

        """
        Calcula a acurácia do modelo com base nas previsões.

        Args:
            output (torch.Tensor): Saída do modelo.
            y_test (torch.Tensor): Rótulos reais.
            n_classes (int): Número de classes.

        Returns:
            float: Acurácia da predição.
        """

        y_test = y_test.float().squeeze()

        if n_classes == 1:
            preds = (torch.sigmoid(output) > 0.5).float()
        else:
            preds = torch.argmax(output, dim=1)
            
        if y_test.ndim > 1:
            y_test = torch.argmax(y_test, dim=1)

        correct = (preds == y_test).float().sum()
        total = y_test.shape[0]

        return (correct / total).item()



    def train_model(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        optimizer: str,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None
        ) -> Tuple[List[float], List[float], List[float], List[float]]:           # definindo a função de treinamento

        """
        Treina o modelo com base nos dados fornecidos.

        Args:
            X_train (torch.Tensor): Dados de entrada de treino.
            X_val (torch.Tensor): Dados de entrada de validação.
            y_train (torch.Tensor): Rótulos de treino.
            y_val (torch.Tensor): Rótulos de validação.
            optimizer (str): Tipo de otimizador ("sgd" ou "adam").

        Returns:
            Tuple[List[float], List[float], List[float], List[float]]:
                - Lista de perdas de treino por época.
                - Lista de perdas de validação por época.
                - Lista de acurácias de treino por época.
                - Lista de acurácias de validação por época.
        """

        # Definindo o otimizador
        if self.weight_decay is not None:
            if optimizer == "sgd":
                self.optimizer = torch.optim.SGD(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
            elif optimizer == "adam":
                self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
            else:
                raise ValueError("Otimizador não reconhecido. Escolha entre 'sgd' ou 'adam'.")
        else:
            if optimizer == "sgd":
                self.optimizer = torch.optim.SGD(self.parameters(), lr = self.lr)
            elif optimizer == "adam":
                self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
            else:
                raise ValueError("Otimizador não reconhecido. Escolha entre 'sgd' ou 'adam'.")

        
        criterion = nn.BCEWithLogitsLoss() if self.n_classes == 1 else nn.CrossEntropyLoss()        # função de perda diferente para classificação binária ou multiclasse 

        if self.n_classes > 1:
            y_train = torch.tensor(y_train).long()
            y_val = torch.tensor(y_val).long() if y_val is not None else None

        loss_train_list = []
        loss_valid_list = []
        accuracy_train_list = []
        accuracy_valid_list = []

        for epoch in range(self.n_epochs):
            self.train()

            # Forward
            output = self.forward(X_train)

            # Loss
            y_train = (y_train + 1) // 2 if self.n_classes == 1 else y_train
            y_val = (y_val + 1) // 2 if self.n_classes == 1 and X_val is not None and y_val is not None else y_val
            y_train = y_train.squeeze() if self.n_classes > 1 else y_train

            loss = criterion(output, y_train)
            loss_train_list.append(loss.item())

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Debug
            # if epoch % 10 == 0 or epoch == self.n_epochs - 1:
            #     self.eval()
            #     with torch.no_grad():
            #         sample_output = self.forward(X_train[:5])
            #         if self.n_classes == 1:
            #             probabilities = torch.sigmoid(sample_output)
            #             preds = (probabilities > 0.5).int().flatten()
            #         else:
            #             probabilities = torch.softmax(sample_output, dim=1)
            #             preds = torch.argmax(probabilities, dim=1)
                    
            #         print("\n=== Verificacao de Amostra ===")
            #         print(f"Epoca {epoch + 1}:")
            #         print("Saida bruta:", sample_output)
            #         print("Probabilidades:", probabilities)
            #         print("Predicoes:", preds)
            #         print("Labels reais:", y_train[:5].flatten())
            #         acc = (preds == y_train[:5].flatten()).float().mean().item()
            #         print(f"Acuracia nesta amostra: {acc:.2%}")

            # Acurácia no treino
            accuracy_train = self.compute_accuracy(output, y_train, self.n_classes)
            accuracy_train_list.append(accuracy_train)

            if X_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    y_val = y_val.squeeze() if self.n_classes > 1 else y_val
                    val_output = self.forward(X_val)
                    val_loss = criterion(val_output, y_val)
                    loss_valid_list.append(val_loss.item())

                    accuracy_valid = self.compute_accuracy(val_output, y_val, self.n_classes)
                    accuracy_valid_list.append(accuracy_valid)

                print(f"Epoca {epoch+1}/{self.n_epochs} - Perda (treino): {loss.item():.4f} - Perda (validacao): {val_loss:.4f} - Acuracia (treino): {accuracy_train:.4f} - Acuracia (validacao): {accuracy_valid:.4f}")

                self.check_early_stop(val_loss.item())

                if self.stop_training:
                    break

            else:
                print(f"Epoca {epoch+1}/{self.n_epochs} - Perda (treino): {loss.item():.4f} - Acuracia (treino): {accuracy_train:.4f}")


        return loss_train_list, loss_valid_list, accuracy_train_list, accuracy_valid_list

        
    def predict(self, X_test: torch.Tensor) -> torch.Tensor:

        """
        Gera previsões a partir dos dados de entrada.

        Coloca o modelo em modo de avaliação, realiza a inferência sem cálculo de gradientes
        e retorna as classes previstas.

        Args:
            X_test (torch.Tensor): Tensor de entrada com shape (batch_size, n_features).

        Returns:
            torch.Tensor: Previsões do modelo.
                - Para classificação binária: tensor de rótulos 0 ou 1 (shape: [batch_size]).
                - Para classificação multiclasse: tensor com os índices da classe prevista (shape: [batch_size]).
        """

        self.eval()
        with torch.no_grad():
            output = self.forward(X_test)
            if self.n_classes == 1:
                return (torch.sigmoid(output) > 0.5).float()
            else:
                return torch.argmax(output, dim=1)




    def evaluate_accuracy(self, X_test: torch.Tensor, y_test: torch.Tensor) -> float:

        """
        Calcula a acurácia do modelo em um conjunto de dados não visto.

        Args:
            X_test (torch.Tensor): Dados de entrada.
            y_test (torch.Tensor): Rótulos reais.

        Returns:
            float: Acurácia da predição no intervalo [0, 1].
        """

        y_pred = self.predict(X_test)
        correct = (y_pred == y_test).float().sum()
        return (correct / y_test.shape[0]).item()