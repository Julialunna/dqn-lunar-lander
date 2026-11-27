"""Núcleo compartilhado pelos experimentos: QNetwork e DQNAgent (cópia do notebook)
e funções auxiliares de semente, avaliação e treino."""
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ENV_ID = "LunarLander-v3"
SEED_TESTE = 100        # sementes 100..199: as mesmas usadas no relatório
SEED_VALIDACAO = 1000   # sementes 1000..1099: conjunto novo, nunca usado na busca

CONFIG_BASE = {
    "alpha": 0.0003,
    "gamma": 0.99,
    "buffer_size": 50_000,
    "batch_size": 64,
    "target_update_interval": 1_000,
    "epsilon_initial": 1.0,
    "epsilon_decay": 0.99995,
    "epsilon_min": 0.05,
    "episodes": 1_000,
}


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


class DQNAgent:
    def __init__(
        self,
        observation_space,
        action_space,
        alpha,
        gamma,
        buffer_size,
        batch_size,
        target_update_interval,
        epsilon_initial,
        epsilon_decay,
        epsilon_min,
        device="cpu",
    ):
        self.observation_space = observation_space
        self.action_space = action_space

        self.state_dim = observation_space.shape[0]
        self.num_actions = action_space.n

        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval

        self.epsilon = epsilon_initial
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.buffer_size = buffer_size
        self._states = np.empty((buffer_size, self.state_dim), dtype=np.float32)
        self._actions = np.empty(buffer_size, dtype=np.int64)
        self._rewards = np.empty(buffer_size, dtype=np.float32)
        self._next_states = np.empty((buffer_size, self.state_dim), dtype=np.float32)
        self._terminals = np.empty(buffer_size, dtype=np.float32)
        self._buf_ptr = 0
        self._buf_size = 0

        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(1)

        self.action_value = QNetwork(self.state_dim, self.num_actions).to(self.device)
        self.target = QNetwork(self.state_dim, self.num_actions).to(self.device)
        self.target.load_state_dict(self.action_value.state_dict())
        self.target.requires_grad_(False)
        self.target.eval()

        self.optimizer = optim.Adam(self.action_value.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()
        self.current_step = 0

    def choose_action(self, state, evaluate=False):
        if evaluate or random.random() >= self.epsilon:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_values = self.action_value(state_tensor)
            return q_values.argmax().item()

        return self.action_space.sample()

    def learn(self):
        if self._buf_size < self.batch_size:
            return
        idx = np.random.randint(0, self._buf_size, size=self.batch_size)

        states = torch.from_numpy(self._states[idx]).to(self.device)
        actions = torch.from_numpy(self._actions[idx]).to(self.device)
        rewards = torch.from_numpy(self._rewards[idx]).to(self.device)
        next_states = torch.from_numpy(self._next_states[idx]).to(self.device)
        terminals = torch.from_numpy(self._terminals[idx]).to(self.device)

        predictions = self.action_value(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target(next_states).max(dim=1).values
            targets = rewards + self.gamma * max_next_q * (1.0 - terminals)

        loss = self.loss_fn(predictions, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

    def train(self, env, episodes, verbose=True):
        self.training_returns = []
        rewards = self.training_returns

        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                self._states[self._buf_ptr] = state
                self._actions[self._buf_ptr] = action
                self._rewards[self._buf_ptr] = reward
                self._next_states[self._buf_ptr] = next_state
                self._terminals[self._buf_ptr] = float(terminated)
                self._buf_ptr = (self._buf_ptr + 1) % self.buffer_size
                if self._buf_size < self.buffer_size:
                    self._buf_size += 1
                self.learn()

                self.current_step += 1
                if self.current_step % self.target_update_interval == 0:
                    self.target.load_state_dict(self.action_value.state_dict())

                self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

                state = next_state
                done = terminated or truncated
                total_reward += reward

            rewards.append(total_reward)

            if verbose and ((episode + 1) % 50 == 0 or episode + 1 == episodes):
                recent_mean = np.mean(rewards[-100:])
                print(
                    f"  Episódio {episode + 1}/{episodes} | "
                    f"média dos últimos {min(100, len(rewards))}: {recent_mean:.1f} | "
                    f"epsilon: {self.epsilon:.3f} | passos: {self.current_step}",
                    flush=True,
                )

        return rewards

    def evaluate(self, env, episodes=100, seed=None):
        was_training = self.action_value.training
        self.action_value.eval()
        rewards = []

        try:
            for episode in range(episodes):
                ep_seed = seed + episode if seed is not None else None
                state, _ = env.reset(seed=ep_seed)
                done = False
                total_reward = 0

                while not done:
                    action = self.choose_action(state, evaluate=True)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    state = next_state
                    done = terminated or truncated
                    total_reward += reward

                rewards.append(total_reward)
        finally:
            if was_training:
                self.action_value.train()

        return rewards

    def save(self, path):
        torch.save(self.action_value.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.action_value.load_state_dict(state_dict)
        self.target.load_state_dict(state_dict)


# ----------------------------------------------------------------------------
# Funções auxiliares
# ----------------------------------------------------------------------------
def fixar_sementes(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def avaliar_politica(escolher_acao, seed_inicial, episodios=100):
    """Avalia qualquer política (função estado -> ação, sem exploração) em
    `episodios` episódios com sementes seed_inicial, seed_inicial+1, ..."""
    env = gym.make(ENV_ID)
    retornos = []
    try:
        for i in range(episodios):
            estado, _ = env.reset(seed=seed_inicial + i)
            terminou, total = False, 0.0
            while not terminou:
                estado, r, term, trunc, _ = env.step(int(escolher_acao(estado)))
                terminou = term or trunc
                total += r
            retornos.append(total)
    finally:
        env.close()
    return np.asarray(retornos, dtype=float)


def politica_de_pt(caminho):
    """Carrega um state_dict da QNetwork (.pt) e devolve a política gulosa."""
    env = gym.make(ENV_ID)
    rede = QNetwork(env.observation_space.shape[0], env.action_space.n)
    env.close()
    rede.load_state_dict(torch.load(caminho, map_location="cpu", weights_only=True))
    rede.eval()

    def escolher(estado):
        with torch.no_grad():
            return rede(torch.as_tensor(estado, dtype=torch.float32)).argmax().item()

    return escolher


def resumo(retornos):
    r = np.asarray(retornos, dtype=float)
    return {
        "media": float(r.mean()),
        "desvio": float(r.std()),
        "mediana": float(np.median(r)),
        "min": float(r.min()),
        "max": float(r.max()),
        "sucesso": float(np.mean(r >= 200) * 100),
    }


def treinar_dqn(params, seed):
    """Reproduz exatamente o protocolo de `rodar_experimento` do notebook
    (mesma ordem de sementes), de modo que seed=42 reproduza a busca original."""
    params = dict(params)
    episodios = params.pop("episodes")
    fixar_sementes(seed)

    env = gym.make(ENV_ID)
    env.action_space.seed(seed)
    env.reset(seed=seed)

    inicio = time.time()
    try:
        agente = DQNAgent(env.observation_space, env.action_space, **params)
        retornos_treino = np.asarray(agente.train(env, episodes=episodios), dtype=float)
    finally:
        env.close()
    return agente, retornos_treino, time.time() - inicio
