"""EXPERIMENTO 2 — Isolar 'implementação' de 'hiperparâmetros' no SB3.

Treina o DQN do Stable Baselines3 com uma configuração equivalente à do nosso agente.
Variantes:
  nossa_cfg            : train_freq=1, alvo a cada 1000 passos, lr=3e-4, batch 64, rede 128x128,
                         gamma=0.995, buffer 50k, epsilon 1.0 -> 0.05 em ~60 mil passos
  nossa_cfg_trainfreq4 : igual, mas train_freq=4 (testa a hipótese de 'menos atualizações por passo')
Diferenças que NÃO são removidas (são do SB3): perda de Huber e clipping de gradiente (norma 10),
decaimento linear (e não geométrico) de epsilon.

Uso:
    python exp2_sb3_nossa_config.py                       # nossa_cfg, seed 42
    python exp2_sb3_nossa_config.py --variantes nossa_cfg nossa_cfg_trainfreq4 --seeds 42 43 44
"""
import argparse
import json
import math
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from dqn_core import ENV_ID, SEED_TESTE, SEED_VALIDACAO, avaliar_politica, resumo

PASSOS_PADRAO = 644_946  # mesmo número de passos do nosso agente final


def kwargs_variante(nome, passos):
    # nosso epsilon: 0.99995^t = 0.05 em t ~ 59.9 mil passos -> fração equivalente no decaimento linear do SB3
    frac = math.log(0.05) / math.log(0.99995) / passos
    kw = dict(
        learning_rate=3e-4,
        buffer_size=50_000,
        learning_starts=64,
        batch_size=64,
        tau=1.0,
        gamma=0.995,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=frac,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    if nome == "nossa_cfg_trainfreq4":
        kw["train_freq"] = 4
    elif nome != "nossa_cfg":
        raise ValueError(f"variante desconhecida: {nome}")
    return kw


def rodar(nome, seed, passos, saida):
    arq = saida / f"exp2_sb3_{nome}_s{seed}.json"
    if arq.exists():
        print(f"[pulando] {arq.name} já existe")
        return
    print(f"\n=== SB3 {nome} | seed {seed} | {passos} passos ===", flush=True)
    env = Monitor(gym.make(ENV_ID))
    env.reset(seed=seed)
    inicio = time.time()
    try:
        modelo = DQN("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **kwargs_variante(nome, passos))
        modelo.learn(total_timesteps=passos, log_interval=50)
        treino = np.asarray(env.get_episode_rewards(), dtype=float)
    finally:
        env.close()
    tempo = time.time() - inicio
    modelo.save(str(saida / f"exp2_sb3_{nome}_s{seed}"))

    pol = lambda s: modelo.predict(s, deterministic=True)[0]
    teste = avaliar_politica(pol, SEED_TESTE, 100)
    val = avaliar_politica(pol, SEED_VALIDACAO, 100)
    res = {
        "variante": nome, "seed": seed, "passos": passos, "tempo_s": tempo,
        "episodios_treino": len(treino), "train_last100": float(treino[-100:].mean()),
        "teste_100_199": resumo(teste), "validacao_1000_1099": resumo(val),
        "train_returns": treino.tolist(), "test_returns": teste.tolist(), "val_returns": val.tolist(),
    }
    arq.write_text(json.dumps(res, indent=2), encoding="utf-8")
    t, v = res["teste_100_199"], res["validacao_1000_1099"]
    print(f"{nome} s{seed}: treino(100) {res['train_last100']:.1f} | teste {t['media']:.1f} ± {t['desvio']:.1f} "
          f"({t['sucesso']:.0f}%) | validação {v['media']:.1f} ± {v['desvio']:.1f} ({v['sucesso']:.0f}%) | {tempo:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variantes", nargs="+", default=["nossa_cfg"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--passos", type=int, default=PASSOS_PADRAO)
    ap.add_argument("--saida", default="exp2_resultados")
    args = ap.parse_args()
    saida = Path(args.saida)
    saida.mkdir(exist_ok=True)
    for v in args.variantes:
        for s in args.seeds:
            rodar(v, s, args.passos, saida)

    print("\nResumo (referência: SB3 padrão = -3.2 ± 86.1, 7% | nosso DQN = 276.8 ± 24.4, 97%)")
    for arq in sorted(saida.glob("exp2_sb3_*.json")):
        r = json.loads(arq.read_text(encoding="utf-8"))
        t, v = r["teste_100_199"], r["validacao_1000_1099"]
        print(f"  {r['variante']:22s} s{r['seed']:<3d} teste {t['media']:7.1f} ({t['sucesso']:3.0f}%) | "
              f"validação {v['media']:7.1f} ({v['sucesso']:3.0f}%) | {r['tempo_s']:.0f}s")


if __name__ == "__main__":
    main()
