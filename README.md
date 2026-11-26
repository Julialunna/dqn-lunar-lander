# DQN Lunar Lander

Projeto de aprendizado por reforço com **Deep Q-Network (DQN)** para treinar um agente a pousar uma nave no ambiente Lunar Lander. Realizado como Trabalho Prático da disciplina de Aprendizado por Reforço na UFMG em 2026/2.
Uma documentação detalhada está disponível em [Documentacao.pdf](./Documentacao.pdf).

Conteúdo:

0. [Integrantes](#0-integrantes)
1. [Objetivo](#1-objetivo)
2. [Ambiente e implementação](#2-ambiente-e-implementação)
3. [Arquivos disponíveis](#3-arquivos-disponíveis)

## 0. Integrantes
- Carla B. Ferreira
- Gabriel de S. Gomes
- Julia A. Luna
- Professor Orientador: Luiz Chaimowicz

## 1. Objetivo

O agente observa o estado do ambiente, escolhe uma ação e recebe uma recompensa. Ao longo dos episódios, aprende uma política que busca maximizar a recompensa acumulada e concluir um pouso seguro.

## 2. Ambiente e implementação

O Lunar Lander apresenta um espaço de estados contínuo e um conjunto discreto de ações de controle. O DQN utiliza uma rede neural para estimar o valor Q das ações disponíveis e atualizar suas escolhas com base nas recompensas observadas.

Em uma implementação DQN, o treinamento normalmente combina exploração e aproveitamento (política ε-greedy), replay de experiências e uma rede-alvo para tornar o aprendizado mais estável. 

### 2.1 Execução

#### Requisitos
- Python 3.12 ou superior e os pacotes de `requirements.txt`:
  `gymnasium[box2d]==1.4.0`, `torch==2.15.0`, `numpy==2.5.5`, `matplotlib==3.11.2`,
  `ipykernel==7.4.0`, `stable-baselines3==2.9.2`.
- O extra `box2d` pode exigir o `swig` instalado (em Windows, costuma bastar `pip install swig`).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### Rodando a implementação localmente
Toda a execução pode ser realizada a partir do Notebook Python em [dqn_lunar_lander.ipynb](./dqn_lunar_lander.ipynb). Em um terminal, navegue até o diretório deste trabalho e execute o seguinte comando:

```
jupyter notebook
```

Dentro da interface do Jupyter, você verá os arquivos correspondentes listados. Basta abrir o notebook (arquivo .ipynb) e seguir as instruções de forma sequencial. Caso não queira carregar os arquivos e reexecutar os treinamentos do zero, basta apagar os arquivos `.pt`/`.json`. 
Para gerar os gráficos, execute o comando `python gerar_graficos.py`.

## 2.2 Avaliação

A partir do arquivo .ipynb é possível acompanhar as recompensas por episódio e o comportamento do agente ao longo do treinamento. Os resultados podem variar entre execuções devido à exploração e à aleatoriedade do ambiente, assim, foram salvos resultados em arquivos .json e nos gráficos disponíveis em [/graficos/](./graficos/). Uma discussão mais detalhada é apresentada na documentação do trabalho.

## 3. Arquivos disponíveis

| Arquivo | Descrição |
|---|---|
| `dqn_lunar_lander.ipynb` | DQN, treino/teste da referência, busca de hiperparâmetros, treino final e comparação com o SB3 |
| `gerar_graficos.py` | Gera os gráficos do relatório (pasta `graficos/`) a partir dos JSON |
| `dqn_baseline.pt`, `dqn_melhor.pt`, `dqn_sb3.zip` | Modelos treinados (referência, melhor configuração, SB3) |
| `*_resultado.json`, `resultados_busca.json` | Retornos de treino/teste de cada experimento |