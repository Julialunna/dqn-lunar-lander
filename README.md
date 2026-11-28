# DQN Lunar Lander

Trabalho Prático da disciplina de Aprendizado por Reforço da UFMG, 2026/2. O projeto implementa um **Deep Q-Network (DQN)** para o ambiente `LunarLander-v3`, busca hiperparâmetros e compara os resultados com o DQN do Stable Baselines3 (SB3).

A metodologia, os resultados e as conclusões estão no [relatório do trabalho](./_RL__Relatorio_Trabalho_Pratico_Carla_Julia_Gabriel.pdf).

## 1. Integrantes

- Carla B. Ferreira
- Gabriel de S. Gomes
- Julia A. Luna
- Professor orientador: Luiz Chaimowicz

## 2. Instalação

Os experimentos utilizaram Python 3.12 e 3.13, com treinamento em CPU. Execute os comandos abaixo na raiz do projeto, onde estão este README e `requirements.txt`.

No Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

No Windows, em PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Mantenha o ambiente virtual ativado nos comandos das próximas seções.

As dependências estão listadas em [requirements.txt](./requirements.txt):

| Pacote | Versão | Uso |
|---|---|---|
| `gymnasium[box2d]` | 1.4.0 | Ambiente Lunar Lander e física Box2D |
| `torch` | 2.15.0 | Redes neurais e treinamento |
| `numpy` | 2.5.5 | Replay buffer e cálculos numéricos |
| `matplotlib` | 3.11.2 | Gráficos |
| `ipykernel` | 7.4.0 | Kernel Python do notebook |
| `stable-baselines3` | 2.9.2 | Implementação de comparação |
| `notebook` | Conforme instalação | Interface Jupyter no navegador, usada na opção pelo terminal |

Se a instalação do extra Box2D solicitar o SWIG, instale-o com `python -m pip install swig` e repita a instalação das dependências.

## 3. Consultar os resultados salvos

O projeto inclui modelos treinados, retornos por episódio em JSON e gráficos correspondentes às execuções analisadas no relatório.

### 3.1 Notebook principal

O notebook pode ser executado pelo VS Code ou pelo Jupyter iniciado no terminal. Nas duas opções, use o ambiente `.venv` com as dependências instaladas.

#### Opção 1: VS Code

1. Instale as extensões **Python** e **Jupyter** no VS Code.
2. Abra a pasta raiz do projeto no editor e, em seguida, o arquivo [dqn_lunar_lander.ipynb](./dqn_lunar_lander.ipynb).
3. No canto superior direito do notebook, clique em **Select Kernel / Selecionar Kernel**, escolha **Python Environments / Ambientes Python** e selecione o interpretador da `.venv`.
4. Execute as células de cima para baixo ou use **Run All / Executar Tudo**.


#### Opção 2: terminal e navegador

Com a `.venv` ativada, execute na raiz do projeto:

```bash
python -m notebook
```

O notebook contém a implementação própria, o treinamento e a avaliação da referência, a busca de 17 configurações, o treinamento final e a comparação com o SB3 padrão.

Com os modelos e JSON fornecidos presentes, as etapas de treinamento reutilizam os resultados salvos. A avaliação da referência em 100 episódios é executada novamente. Se os arquivos necessários estiverem ausentes, a respectiva etapa realiza um novo treinamento. Mudanças de parâmetros também podem fazer a busca ou o treinamento final ignorarem os resultados salvos.

### 3.2 Gráficos principais

Na raiz do projeto:

```bash
python gerar_graficos.py
```

O comando gera as seis figuras em [graficos/](./graficos/), usando os JSON salvos. Como o JSON da referência contém apenas o histórico de treinamento, o script também avalia `dqn_baseline.pt` em 100 episódios para produzir seu gráfico de teste. Essa etapa executa o ambiente, sem retreinar o agente.

### 3.3 Experimentos adicionais

As instruções completas estão em [experimentos/README_experimentos.md](./experimentos/README_experimentos.md). Os comandos abaixo devem ser executados dentro dessa pasta:

```bash
cd experimentos
```

Para reavaliar os modelos da busca, a referência, o modelo final e o SB3 nas sementes 100–199 e 1000–1099:

```bash
python exp1_validacao.py --dir ..
```

O comando não treina modelos: ele recalcula as avaliações, imprime a comparação e atualiza `exp1_validacao.json`.

Para consultar o resumo das duas variantes ajustadas do SB3:

```bash
python exp2_sb3_nossa_config.py --variantes nossa_cfg nossa_cfg_trainfreq4
```

As execuções com JSON existente são puladas e o resumo é impresso. Uma variante sem resultado salvo é treinada e avaliada.

Para recalcular somente o resumo dos 15 treinamentos já salvos e gerar a figura de múltiplas sementes:

```bash
python exp3_multiplas_sementes.py --so-agregar
```

Esse comando atualiza `experimentos/exp3_resumo.json` e gera `experimentos/graficos/07_multiplas_sementes.png`. Não inicia treinamentos. Ao terminar, retorne à raiz com `cd ..`.

## 4. Executar novos treinamentos

Para repetir o experimento principal do zero preservando os resultados entregues, crie uma pasta separada e copie para ela `dqn_lunar_lander.ipynb` e `requirements.txt`. Use o ambiente virtual com as dependências instaladas, abra o notebook nessa pasta e execute todas as células em ordem. Sem os modelos e JSON anteriores, ele treina a referência, executa a busca, treina a configuração final e treina o SB3 padrão, salvando os novos resultados nessa pasta.

Para repetir os treinamentos adicionais, crie uma pasta separada com os quatro arquivos `.py` de `experimentos/` e, dentro dela, execute:

```bash
python exp2_sb3_nossa_config.py --variantes nossa_cfg nossa_cfg_trainfreq4 --seeds 42
python exp3_multiplas_sementes.py --configs base gamma_0.995 target_2500 --seeds 42 43 44 45 46 --workers 4
```

O segundo comando realiza três configurações × cinco sementes, com 1.000 episódios por treinamento. Ajuste `--workers` conforme a capacidade da máquina. Os scripts salvam os resultados concluídos e pulam essas execuções ao serem rodados novamente.

Fixar as sementes não garante resultados numericamente idênticos entre ambientes de execução. Novos treinamentos podem apresentar diferenças relevantes de desempenho. Os JSON e modelos entregues preservam as execuções discutidas no relatório; os scripts permitem repetir o protocolo experimental.

## 5. Arquivos disponíveis

| Arquivo ou pasta | Conteúdo |
|---|---|
| [Relatório em PDF](./_RL__Relatorio_Trabalho_Pratico_Carla_Julia_Gabriel.pdf) | Documentação do trabalho |
| `dqn_lunar_lander.ipynb` | Implementação própria e experimentos principais |
| `requirements.txt` | Pacotes necessários à execução |
| `gerar_graficos.py`, `graficos/` | Gerador e seis figuras dos experimentos principais |
| `dqn_baseline.pt`, `dqn_melhor.pt`, `dqn_sb3.zip` | Modelos principais treinados |
| `*_resultado.json` | Resultados dos modelos principais |
| `resultados_busca.json`, `modelos_busca/` | Resultados e modelos das 17 configurações da busca |
| `experimentos/dqn_core.py` | Núcleo DQN compartilhado pelos experimentos adicionais |
| `experimentos/exp1_validacao.py`, `experimentos/exp1_validacao.json` | Reavaliação dos modelos em dois conjuntos de sementes |
| `experimentos/exp2_sb3_nossa_config.py`, `experimentos/exp2_resultados/` | Variantes ajustadas do SB3, resultados e modelos |
| `experimentos/exp3_multiplas_sementes.py` | Treinamento e agregação de múltiplas sementes |
| `experimentos/exp3_resultados/`, `experimentos/exp3_modelos/` | Resultados e modelos dos 15 treinamentos adicionais |
| `experimentos/exp3_resumo.json`, `experimentos/graficos/` | Resumo entre sementes e figura adicional |

## 6. Conteúdo da entrega

O ZIP deve incluir o relatório, este README, `requirements.txt`, o notebook, os scripts, os JSON, os gráficos e os modelos treinados, preservando a estrutura de pastas. Inclua também o README de `experimentos/`.

As pastas `.venv/`, `.git/`, `__pycache__/` e `.ipynb_checkpoints/` podem ser excluídas do ZIP. Os arquivos `.zip` do SB3 são modelos treinados e devem permanecer na entrega.
