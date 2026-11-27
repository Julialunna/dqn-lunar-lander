"""EXPERIMENTO 3 — Múltiplas sementes de treino para as configurações finalistas.

Cada (configuração, semente) é um treinamento independente, executado em paralelo
(1 processo por treino; cada treino usa 1 thread de CPU). Resultados parciais ficam em
exp3_resultados/*.json, então o script pode ser interrompido e retomado.

Configurações disponíveis (--configs):
  base            : configuração de referência, 1000 episódios
  gamma_0.995     : gamma=0.995, 1000 episódios (vencedor da busca)
  target_2500     : alvo a cada 2500 passos, 1000 episódios
  gamma_0.995_2000ep : gamma=0.995, 2000 episódios (configuração que foi usada no modelo final)

Uso:
    python exp3_multiplas_sementes.py                                   # 3 configs x 5 sementes
    python exp3_multiplas_sementes.py --seeds 42 43 44 --workers 3
    python exp3_multiplas_sementes.py --configs gamma_0.995 gamma_0.995_2000ep --seeds 42 43 44
    python exp3_multiplas_sementes.py --so-agregar                      # apenas resume o que já existe

Estimativa: ~7-8 min por treino de 1000 episódios em CPU (12 min para 2000). Com 4 processos,
15 treinos levam ~30 min.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from dqn_core import CONFIG_BASE, SEED_TESTE, SEED_VALIDACAO

CONFIGS = {
    "base": {},
    "gamma_0.995": {"gamma": 0.995},
    "target_2500": {"target_update_interval": 2_500},
    "gamma_0.995_2000ep": {"gamma": 0.995, "episodes": 2_000},
}
SAIDA = Path("exp3_resultados")
MODELOS = Path("exp3_modelos")


def _job(nome, seed):
    """Executado em um processo separado (importações dentro da função por causa do 'spawn')."""
    from dqn_core import avaliar_politica, resumo, treinar_dqn

    arq = SAIDA / f"{nome}_s{seed}.json"
    if arq.exists():
        return nome, seed, "pulado"
    params = dict(CONFIG_BASE)
    params.update(CONFIGS[nome])

    agente, treino, tempo = treinar_dqn(params, seed)
    agente.save(str(MODELOS / f"{nome}_s{seed}.pt"))
    pol = lambda s: agente.choose_action(s, evaluate=True)
    teste = avaliar_politica(pol, SEED_TESTE, 100)
    val = avaliar_politica(pol, SEED_VALIDACAO, 100)
    res = {
        "config": nome, "seed": seed, "params": params, "steps": int(agente.current_step),
        "tempo_s": tempo, "train_last100": float(treino[-100:].mean()),
        "teste_100_199": resumo(teste), "validacao_1000_1099": resumo(val),
        "train_returns": treino.tolist(), "test_returns": teste.tolist(), "val_returns": val.tolist(),
    }
    arq.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return nome, seed, f"teste {res['teste_100_199']['media']:.1f} ({res['teste_100_199']['sucesso']:.0f}%) | " \
                       f"validação {res['validacao_1000_1099']['media']:.1f} ({res['validacao_1000_1099']['sucesso']:.0f}%) | {tempo:.0f}s"


def agregar():
    por_config = {}
    for arq in sorted(SAIDA.glob("*.json")):
        r = json.loads(arq.read_text(encoding="utf-8"))
        por_config.setdefault(r["config"], []).append(r)
    if not por_config:
        print("Nenhum resultado em", SAIDA)
        return

    print(f"\n{'Configuração':22s} {'n':>2s} | {'Teste 100-199 (média das sementes ± DP)':>40s} | "
          f"{'Validação 1000-1099':>28s} | {'Sucesso val. (%)':>16s}")
    resumo_final = {}
    for nome, rs in por_config.items():
        mt = np.array([r["teste_100_199"]["media"] for r in rs])
        mv = np.array([r["validacao_1000_1099"]["media"] for r in rs])
        sv = np.array([r["validacao_1000_1099"]["sucesso"] for r in rs])
        st = np.array([r["teste_100_199"]["sucesso"] for r in rs])
        ddof = 1 if len(rs) > 1 else 0
        resumo_final[nome] = {
            "n_sementes": len(rs), "sementes": [r["seed"] for r in rs],
            "teste_media_das_sementes": float(mt.mean()), "teste_dp_entre_sementes": float(mt.std(ddof=ddof)),
            "validacao_media_das_sementes": float(mv.mean()), "validacao_dp_entre_sementes": float(mv.std(ddof=ddof)),
            "sucesso_teste_medio": float(st.mean()), "sucesso_validacao_medio": float(sv.mean()),
            "sucesso_validacao_por_semente": sv.tolist(),
            "medias_validacao_por_semente": mv.tolist(),
            "tempo_medio_s": float(np.mean([r["tempo_s"] for r in rs])),
        }
        print(f"{nome:22s} {len(rs):>2d} | {mt.mean():8.1f} ± {mt.std(ddof=ddof):6.1f} (suc. {st.mean():5.1f}%)      | "
              f"{mv.mean():8.1f} ± {mv.std(ddof=ddof):6.1f}      | {sv.mean():6.1f} [{', '.join(f'{x:.0f}' for x in sv)}]")
    Path("exp3_resumo.json").write_text(json.dumps(resumo_final, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSalvo em exp3_resumo.json (o desvio é ENTRE sementes, com ddof=1).")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        nomes = list(por_config)
        for i, nome in enumerate(nomes):
            ys = [r["validacao_1000_1099"]["media"] for r in por_config[nome]]
            ax.scatter(np.full(len(ys), i) + np.linspace(-0.08, 0.08, len(ys)), ys, alpha=0.8)
            ax.hlines(np.mean(ys), i - 0.25, i + 0.25, color="black")
        ax.axhline(200, color="tab:green", linestyle=":", label="Meta (200)")
        ax.set_xticks(range(len(nomes)))
        ax.set_xticklabels(nomes, rotation=15)
        ax.set_ylabel("Retorno médio (validação, 100 ep.)")
        ax.set_title("Variabilidade entre sementes de treino (um ponto por semente)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        Path("graficos").mkdir(exist_ok=True)
        fig.savefig("graficos/07_multiplas_sementes.png", dpi=150, bbox_inches="tight")
        print("Gráfico salvo em graficos/07_multiplas_sementes.png")
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["base", "gamma_0.995", "target_2500"], choices=list(CONFIGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--so-agregar", action="store_true")
    args = ap.parse_args()

    SAIDA.mkdir(exist_ok=True)
    MODELOS.mkdir(exist_ok=True)

    if not args.so_agregar:
        jobs = [(c, s) for s in args.seeds for c in args.configs]
        print(f"{len(jobs)} treinos, {args.workers} processos em paralelo", flush=True)
        # 'spawn' funciona igual em Windows, Linux e macOS
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_job, c, s): (c, s) for c, s in jobs}
            for f in as_completed(futs):
                try:
                    nome, seed, msg = f.result()
                    print(f"[ok] {nome} s{seed}: {msg}", flush=True)
                except Exception as e:  # um treino falho não derruba os outros
                    print(f"[erro] {futs[f]}: {e!r}", flush=True)
    agregar()


if __name__ == "__main__":
    main()
