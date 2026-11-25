import json
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, state):
        return self.layers(state)


def media_movel(valores, janela=50):
    w = min(janela, len(valores))
    return np.convolve(valores, np.ones(w) / w, mode="valid"), w


def avaliar_baseline(model_path: Path, episodes: int = 100, seed: int = 100) -> np.ndarray:
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    net = QNetwork(state_dim, num_actions)
    net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    net.eval()

    rewards = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            done = False
            total_reward = 0.0
            while not done:
                state_tensor = torch.as_tensor(state, dtype=torch.float32)
                with torch.no_grad():
                    action = net(state_tensor).argmax().item()
                next_state, reward, terminated, truncated, _ = env.step(action)
                state = next_state
                done = terminated or truncated
                total_reward += reward
            rewards.append(total_reward)
    finally:
        env.close()

    return np.asarray(rewards, dtype=float)


def main():
    out_dir = Path("graficos")
    out_dir.mkdir(exist_ok=True)

    dados_baseline = json.loads(Path("dqn_baseline_resultado.json").read_text(encoding="utf-8"))
    returns = np.asarray(dados_baseline["returns"], dtype=float)

    window = min(50, len(returns))
    episodes = np.arange(1, len(returns) + 1)
    moving_average = np.convolve(returns, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes, returns, alpha=0.3, label="Retorno por episódio")
    ax.plot(episodes[window - 1:], moving_average, label=f"Média móvel de {window} episódios")
    ax.set_title("DQN no Lunar Lander — treinamento")
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Retorno acumulado por episódio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "01_baseline_treinamento.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if "test_returns" in dados_baseline:
        test_returns = np.asarray(dados_baseline["test_returns"], dtype=float)
    else:
        test_returns = avaliar_baseline(Path("dqn_baseline.pt"), episodes=100, seed=100)

    success_rate = np.mean(test_returns >= 200) * 100
    print(f"Baseline teste -> Episódios: {len(test_returns)}")
    print(f"Baseline teste -> Média: {test_returns.mean():.2f} ± {test_returns.std():.2f}")
    print(f"Baseline teste -> Mediana: {np.median(test_returns):.2f}")
    print(f"Baseline teste -> Mín / Máx: {test_returns.min():.2f} / {test_returns.max():.2f}")
    print(f"Baseline teste -> Taxa de sucesso (>= 200): {success_rate:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    test_episodes = np.arange(1, len(test_returns) + 1)
    axes[0].plot(test_episodes, test_returns, marker="o", markersize=3, alpha=0.7, label="Retorno")
    axes[0].axhline(test_returns.mean(), color="tab:orange", linestyle="--", label=f"Média ({test_returns.mean():.1f})")
    axes[0].axhline(200, color="tab:green", linestyle=":", label="Meta (200)")
    axes[0].set_title("DQN no Lunar Lander — teste")
    axes[0].set_xlabel("Episódio de teste")
    axes[0].set_ylabel("Retorno acumulado")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(test_returns, bins=15, edgecolor="black", alpha=0.75)
    axes[1].axvline(test_returns.mean(), color="tab:orange", linestyle="--", label=f"Média ({test_returns.mean():.1f})")
    axes[1].axvline(200, color="tab:green", linestyle=":", label="Meta (200)")
    axes[1].set_title("Distribuição dos retornos no teste")
    axes[1].set_xlabel("Retorno acumulado")
    axes[1].set_ylabel("Frequência")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "02_baseline_teste.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    ordem_experimentos = [
        "base_1000ep",
        "alpha_0.0001",
        "alpha_0.001",
        "gamma_0.95",
        "gamma_0.995",
        "buffer_10k",
        "buffer_100k",
        "batch_32",
        "batch_128",
        "target_250",
        "target_2500",
        "eps_decay_0.9995",
        "eps_decay_0.99998",
        "eps_min_0.01",
        "eps_init_0.5",
        "ep_500",
        "ep_2000",
    ]
    dados_busca = json.loads(Path("resultados_busca.json").read_text(encoding="utf-8"))
    resultados_busca = []
    for nome in ordem_experimentos:
        if nome in dados_busca:
            entry = dict(dados_busca[nome])
            entry["train_returns"] = np.asarray(entry["train_returns"], dtype=float)
            entry["test_returns"] = np.asarray(entry["test_returns"], dtype=float)
            resultados_busca.append(entry)

    grupos = ["alpha", "gamma", "buffer_size", "batch_size", "target_update", "epsilon", "episodes"]
    base_res = resultados_busca[0]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes_flat = axes.flat

    for ax, grupo in zip(axes_flat, grupos):
        mm_base, w_base = media_movel(base_res["train_returns"])
        ax.plot(
            np.arange(w_base, len(base_res["train_returns"]) + 1),
            mm_base,
            linestyle="--",
            color="black",
            label=base_res["nome"],
        )

        for r in resultados_busca:
            if r["grupo"] == grupo:
                mm, w = media_movel(r["train_returns"])
                ax.plot(np.arange(w, len(r["train_returns"]) + 1), mm, label=r["nome"])

        ax.axhline(200, color="tab:green", linestyle=":", alpha=0.7)
        ax.set_title(f"Impacto de {grupo} no treinamento")
        ax.set_xlabel("Episódio")
        ax.set_ylabel("Média móvel (50 ep.)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes_flat[7].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "03_busca_curvas_treinamento.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    nomes = [r["nome"] for r in resultados_busca]
    medias_teste = [r["test_mean"] for r in resultados_busca]
    desvios_teste = [r["test_std"] for r in resultados_busca]
    taxas_sucesso = [r["success_rate"] for r in resultados_busca]
    y_pos = np.arange(len(nomes))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    axes[0].barh(y_pos, medias_teste, xerr=desvios_teste, alpha=0.8, capsize=3)
    axes[0].axvline(200, color="tab:green", linestyle=":", label="Meta (200)")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(nomes)
    axes[0].invert_yaxis()
    axes[0].set_title("Retorno médio no teste (± desvio padrão)")
    axes[0].set_xlabel("Retorno acumulado")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="x")

    axes[1].barh(y_pos, taxas_sucesso, color="tab:orange", alpha=0.8)
    axes[1].set_title("Taxa de sucesso no teste (>= 200)")
    axes[1].set_xlabel("Taxa de sucesso (%)")
    axes[1].set_xlim(0, 105)
    axes[1].grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(out_dir / "04_busca_comparacao_teste.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    res_melhor = json.loads(Path("dqn_melhor_resultado.json").read_text(encoding="utf-8"))
    res_melhor["train_returns"] = np.asarray(res_melhor["train_returns"], dtype=float)
    melhor_test_returns = np.asarray(res_melhor["test_returns"], dtype=float)

    dados_sb3 = json.loads(Path("dqn_sb3_resultado.json").read_text(encoding="utf-8"))
    sb3_train_returns = np.asarray(dados_sb3["train_returns"], dtype=float)
    sb3_test_returns = np.asarray(dados_sb3["test_returns"], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))

    mm_nosso, w1 = media_movel(res_melhor["train_returns"], janela=50)
    ax.plot(np.arange(w1, len(res_melhor["train_returns"]) + 1), mm_nosso, label="Nosso DQN (média móvel 50 ep.)")

    if len(sb3_train_returns) >= 50:
        mm_sb3, w2 = media_movel(sb3_train_returns, janela=50)
        ax.plot(np.arange(w2, len(sb3_train_returns) + 1), mm_sb3, label="SB3 DQN padrão (média móvel 50 ep.)")
    else:
        ax.plot(np.arange(1, len(sb3_train_returns) + 1), sb3_train_returns, label="SB3 DQN padrão")

    ax.axhline(200, color="tab:green", linestyle=":", label="Meta (200)")
    ax.set_title("Comparação no treinamento — Nosso DQN vs. Stable Baselines3")
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Retorno acumulado")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "05_comparacao_sb3_treinamento.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ep_teste = np.arange(1, 101)
    axes[0].plot(ep_teste, melhor_test_returns, marker="o", markersize=3, alpha=0.7, label="Nosso DQN")
    axes[0].plot(ep_teste, sb3_test_returns, marker="x", markersize=3, alpha=0.7, label="SB3 padrão")
    axes[0].axhline(200, color="tab:green", linestyle=":", label="Meta (200)")
    axes[0].set_title("Retorno por episódio de teste")
    axes[0].set_xlabel("Episódio de teste")
    axes[0].set_ylabel("Retorno acumulado")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot([melhor_test_returns, sb3_test_returns], tick_labels=["Nosso DQN", "SB3 padrão"])
    axes[1].axhline(200, color="tab:green", linestyle=":", label="Meta (200)")
    axes[1].set_title("Distribuição dos retornos no teste")
    axes[1].set_ylabel("Retorno acumulado")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "06_comparacao_sb3_teste.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Gráficos salvos em:", sorted(p.name for p in out_dir.glob("*.png")))


if __name__ == "__main__":
    main()
