# Experimentos adicionais

Instale primeiro as dependências conforme o [README principal](../README.md) e mantenha o ambiente virtual ativado. Todos os comandos deste documento devem ser executados dentro de `experimentos/`. A partir da raiz do projeto:

```bash
cd experimentos
```

## 1. Validação em novas sementes

```bash
python exp1_validacao.py --dir ..
```

O script carrega os modelos de `../modelos_busca/` e os modelos principais da raiz, avalia cada política sem exploração nas sementes 100–199 e 1000–1099 e calcula a correlação entre os rankings da busca. Ele não retreina agentes e atualiza `exp1_validacao.json` com os resultados da nova avaliação.

## 2. SB3 com configuração ajustada

```bash
python exp2_sb3_nossa_config.py --variantes nossa_cfg nossa_cfg_trainfreq4 --seeds 42
```

As duas variantes usam rede 128 × 128, taxa de aprendizado de 0,0003, batch 64, fator de desconto 0,995, buffer de 50 mil transições e atualização da rede alvo a cada 1.000 passos. Elas diferem na frequência de atualização da rede principal: uma atualização por passo (`nossa_cfg`) ou a cada quatro passos (`nossa_cfg_trainfreq4`). O orçamento solicitado é de 644.946 passos.

Permanecem a perda de Huber e o recorte de gradiente do SB3. A exploração tem os mesmos valores inicial e mínimo e tempo semelhante para chegar ao mínimo, mas o decaimento é linear, enquanto o DQN próprio usa decaimento multiplicativo.

Os resultados e modelos são salvos em `exp2_resultados/`. Se o JSON de uma combinação de variante e semente já existir, essa execução é pulada. Com os arquivos entregues presentes, o comando consulta o resumo sem treinar novamente.

Para realizar novas execuções das duas variantes sem substituir as entregues, use uma pasta de saída nova:

```bash
python exp2_sb3_nossa_config.py --variantes nossa_cfg nossa_cfg_trainfreq4 --seeds 42 --saida exp2_nova_execucao
```

## 3. Múltiplas sementes de treinamento

Para recalcular apenas o resumo e o gráfico dos resultados entregues:

```bash
python exp3_multiplas_sementes.py --so-agregar
```

Esse comando lê `exp3_resultados/`, atualiza `exp3_resumo.json` e gera `graficos/07_multiplas_sementes.png`, sem iniciar treinamentos. O desvio padrão do resumo é calculado entre as médias de cada semente, com `ddof=1`.

O protocolo dos 15 treinamentos salvos é:

```bash
python exp3_multiplas_sementes.py --configs base gamma_0.995 target_2500 --seeds 42 43 44 45 46 --workers 4
```

Cada treinamento usa 1.000 episódios. As configurações são a referência, a referência com `gamma=0.995` e a referência com atualização da rede alvo a cada 2.500 passos. Os modelos ficam em `exp3_modelos/` e os retornos por episódio em `exp3_resultados/`. Execuções com JSON existente são puladas; as demais são treinadas em processos separados. Ajuste `--workers` conforme a capacidade da máquina.

Para repetir todos os treinamentos preservando os resultados originais, copie os quatro arquivos `.py` de `experimentos/` para uma pasta separada, entre nela e execute o comando acima. O script cria novas pastas de resultados e modelos no diretório de execução.

O script também oferece a configuração `gamma_0.995_2000ep`, para uma extensão com 2.000 episódios. Ela não faz parte dos 15 treinamentos adicionais entregues.

## 4. Reprodução dos resultados

Os scripts usam o núcleo DQN de `dqn_core.py` e fixam as sementes dos geradores aleatórios e do ambiente. Isso permite repetir o protocolo, mas não garante os mesmos valores entre ambientes de execução.

Na repetição com a semente 42, a referência obteve média de teste de 220,4, enquanto a busca original registrou 274,3. O relatório discute essa divergência; sua causa não foi confirmada. Portanto, diferenças relevantes podem ocorrer mesmo com a mesma semente. Os JSON e checkpoints entregues correspondem às execuções analisadas no relatório.

Os tempos variam conforme a máquina e a concorrência entre processos. Os treinamentos adicionais foram executados em paralelo, de modo que seus tempos não devem ser comparados diretamente com os dos experimentos principais.
