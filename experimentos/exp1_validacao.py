"""EXPERIMENTO 1 — Validação em sementes novas (sem retreinar).

Reavalia todos os modelos salvos (busca, baseline, melhor, SB3) em dois conjuntos:
  - sementes 100..199   (as do relatório; usadas na seleção da busca)
  - sementes 1000..1099 (novas)
e mede se o ranking da busca se mantém e quanto a seleção no conjunto de teste foi otimista.

Uso (na pasta com os .pt/.json; coloque a pasta modelos_busca/ ao lado):
    python exp1_validacao.py --dir ..
Saída: exp1_validacao.json e uma tabela no terminal.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from dqn_core import SEED_TESTE, SEED_VALIDACAO, avaliar_politica, politica_de_pt, resumo


def ranks(v):
    return np.argsort(np.argsort(-np.asarray(v))).astype(float)  # 0 = melhor


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="pasta com .pt, .json, dqn_sb3.zip e modelos_busca/")
    ap.add_argument("-n", type=int, default=100, help="episódios por conjunto de sementes")
    args = ap.parse_args()
    base = Path(args.dir)

    modelos = []  # (nome, tipo, caminho, é_da_busca)
    busca = json.loads((base / "resultados_busca.json").read_text(encoding="utf-8"))
    for nome in busca:
        p = base / "modelos_busca" / f"{nome}.pt"
        if p.exists():
            modelos.append((nome, "pt", p, True))
        else:
            print(f"[aviso] modelo da busca não encontrado: {p}")
    for nome, arq in [("baseline_2000ep", "dqn_baseline.pt"), ("dqn_melhor_2000ep", "dqn_melhor.pt")]:
        if (base / arq).exists():
            modelos.append((nome, "pt", base / arq, False))
    if (base / "dqn_sb3.zip").exists():
        modelos.append(("sb3_padrao", "sb3", base / "dqn_sb3", False))

    saida = {}
    for nome, tipo, caminho, _ in modelos:
        if tipo == "pt":
            pol = politica_de_pt(caminho)
        else:
            from stable_baselines3 import DQN as SB3_DQN
            modelo = SB3_DQN.load(str(caminho), device="cpu")
            pol = lambda s, m=modelo: m.predict(s, deterministic=True)[0]
        ret_teste = avaliar_politica(pol, SEED_TESTE, args.n)
        ret_val = avaliar_politica(pol, SEED_VALIDACAO, args.n)
        saida[nome] = {"teste_100_199": resumo(ret_teste), "validacao_1000_1099": resumo(ret_val),
                       "retornos_teste": ret_teste.tolist(), "retornos_validacao": ret_val.tolist()}
        t, v = saida[nome]["teste_100_199"], saida[nome]["validacao_1000_1099"]
        print(f"{nome:20s} teste {t['media']:7.1f} ± {t['desvio']:6.1f} ({t['sucesso']:3.0f}%) | "
              f"validação {v['media']:7.1f} ± {v['desvio']:6.1f} ({v['sucesso']:3.0f}%)", flush=True)

    # Consistência com os JSON do relatório (a reavaliação deve reproduzir os números antigos)
    print("\nChecagem de reprodutibilidade (sementes 100..199 vs. JSON original):")
    for nome in busca:
        if nome in saida:
            antigo, novo = busca[nome]["test_mean"], saida[nome]["teste_100_199"]["media"]
            print(f"  {nome:20s} original {antigo:7.2f} | reavaliado {novo:7.2f} | dif {novo - antigo:+.3f}")

    # Ranking da busca nos dois conjuntos
    nomes_busca = [n for n in busca if n in saida]
    if len(nomes_busca) >= 3:
        mt = [saida[n]["teste_100_199"]["media"] for n in nomes_busca]
        mv = [saida[n]["validacao_1000_1099"]["media"] for n in nomes_busca]
        rho = spearman(mt, mv)
        melhor_t = nomes_busca[int(np.argmax(mt))]
        melhor_v = nomes_busca[int(np.argmax(mv))]
        print(f"\nCorrelação de Spearman entre rankings (teste vs. validação): {rho:.2f}")
        print(f"Melhor na média de teste: {melhor_t} | melhor na média de validação: {melhor_v}")
        print(f"Média das configurações — teste: {np.mean(mt):.1f} | validação: {np.mean(mv):.1f}")
        saida["_ranking"] = {"spearman": rho, "melhor_teste": melhor_t, "melhor_validacao": melhor_v}

    Path("exp1_validacao.json").write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSalvo em exp1_validacao.json")


if __name__ == "__main__":
    main()
