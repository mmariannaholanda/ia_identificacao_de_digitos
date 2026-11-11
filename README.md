# Reconhecimento de dígitos com PyTorch

Rede neural que classifica imagens de dígitos escritos à mão (0 a 9), usando o conjunto MNIST. Projeto de estudo da minha trilha de IA até visão computacional.

O projeto avança em duas etapas: primeiro uma rede densa (MLP) em PyTorch, e depois uma rede convolucional (CNN) com kernels, para comparar as duas nos mesmos dados e com as mesmas métricas.

## Resultados

Ainda em andamento. A tabela será preenchida conforme eu medir cada modelo.

| Modelo | Acurácia no treino | Acurácia no teste |
|---|---|---|
| Baseline (sempre o dígito mais comum) | — | — |
| MLP (784 → 128 → 10) | — | — |
| CNN | — | — |

## Dados

- **Conjunto:** MNIST, de LeCun, Cortes e Burges, carregado pelo `torchvision`, que baixa os arquivos na primeira execução.
- **Conteúdo:** imagens de 28×28 pixels em tons de cinza (0 a 255), cada uma com o dígito certo como rótulo.
- **Divisão:** 60000 imagens de treino e 10000 de teste, já separadas pelo próprio conjunto (`train=True` e `train=False`).

## Como funciona

**Padronização.** Calculo uma média e um desvio únicos, usando todos os pixels de todas as imagens de treino, e padronizo com `(pixel - média) / desvio`. O teste usa a média e o desvio do treino, para não vazar informação.

**Baseline.** Conto qual dígito é o mais frequente no treino e meço a acurácia de quem sempre chuta esse dígito no teste. Uma rede só tem valor se passar desse número.

**Entradas.** No MLP, cada imagem de 28×28 é achatada em uma fila de 784 números.

**Arquitetura do MLP.**

```
784 entradas → 128 neurônios (ReLU) → 10 notas (uma por dígito)
```

São 101.770 parâmetros. A saída fica sem ativação, porque o softmax já acontece dentro da função de perda. O número de neurônios (128) é um ponto de partida, e ainda não foi validado.

**Treino.**

- Perda: entropia cruzada multiclasse (`CrossEntropyLoss`)
- Otimizador: gradiente descendente (`SGD`), taxa de aprendizado 0,1
- Semente fixa (`torch.manual_seed(42)`) para o sorteio dos pesos ser reproduzível
- Cada passo usa as 60000 imagens de uma vez, sem mini-batches por enquanto

**Avaliação.** Acurácia no treino e no teste, comparadas com o baseline.

## Do NumPy ao PyTorch

Este projeto continua o [classificador de tumores](https://github.com/mmariannaholanda/ia_identificacao_tumores), feito em NumPy puro. A estrutura é a mesma, e o PyTorch assume a parte dos gradientes:

| NumPy (tumores) | PyTorch (dígitos) |
|---|---|
| `propagar` | `modelo(entradas)` |
| `calcular_perda` | `CrossEntropyLoss` |
| `calcular_gradientes` | `perda.backward()` |
| `atualizar_parametros` | `otimizador.step()` |

## Próximos passos

- [x] Carregar o MNIST e padronizar
- [x] Baseline
- [x] Montar o MLP
- [ ] Treinar o MLP e medir treino e teste
- [ ] Gráfico da perda
- [ ] Matriz de confusão 10×10 e exemplos de imagens que a rede errou
- [ ] Treinar com mini-batches
- [ ] Separar um conjunto de validação para escolher o número de neurônios e a taxa de aprendizado
- [ ] Construir a CNN com kernels e comparar com o MLP
- [ ] Salvar os pesos (`.npz`)
- [ ] Testar com dígitos desenhados ou fotografados por mim

## Como rodar

```bash
pip install torch torchvision matplotlib jupyter
jupyter notebook reconhecimento_digitos.ipynb
```

No Google Colab, o `torch` e o `torchvision` já vêm instalados. Rode as células de cima para baixo.

## Autora

**Marianna Holanda**, estudante de Engenharia da Computação na UNIFOR. Tenho interesse em visão computacional e desenvolvimento de software, e também em firmware, IoT e Edge AI.

[GitHub](https://github.com/mmariannaholanda)
