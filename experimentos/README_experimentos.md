# Experimentos adicionais

| # | Script | Tempo aprox. | O que responde |
|---|---|---|---|
| 1 | `python exp1_validacao.py --dir ..` | ~5 min (sem treino) | O ranking da busca se mantém em sementes novas (1000–1099)? Qual o viés de selecionar no teste? |
| 2 | `python exp2_sb3_nossa_config.py --variantes nossa_cfg nossa_cfg_trainfreq4` | ~6 min por treino | O SB3 chega perto do nosso DQN com a nossa configuração? A causa é `train_freq`? |
| 3 | `python exp3_multiplas_sementes.py` | ~30 min com 4 processos | Qual a variabilidade entre sementes de treino? 100% vs 97% é diferença real? |

Todos os scripts são retomáveis (pulam resultados já salvos) e usam `dqn_core.py`, que contém o `DQNAgent`
idêntico ao do notebook. Com a semente 42 o `exp3` deve reproduzir os números da busca original
(`base`: 274,3 / 100%; `gamma_0.995`: 279,4 / 100%; `target_2500`: 279,5 / 97%); o `exp1` imprime
uma checagem de reprodutibilidade com os JSON antigos. Pequenas diferenças são possíveis se a máquina
ou as versões das bibliotecas forem diferentes.

Para o item 4 (1000 vs 2000 episódios com `gamma=0.995`):
`python exp3_multiplas_sementes.py --configs gamma_0.995 gamma_0.995_2000ep --seeds 42 43 44`
