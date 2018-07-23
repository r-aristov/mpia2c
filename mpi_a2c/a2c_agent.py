import torch
from torch.distributions import Categorical
from mpi_a2c.mpi_a2c import Replay


class A2CAgent:
    def __init__(self):
        self.__iter = 0
        self.last_value = 0.0
        self.replay = Replay()
        self.total_reward = 0.0
        self.deterministic = False

    def decide(self, x):
        self.replay.states.append(x)
        probs, value = self.forward(x)

        distrib = Categorical(probs)
        action = distrib.sample().item() if not self.deterministic else torch.argmax(probs)
        self.replay.actions.append(action)
        self.last_value = value.item()
        self.__iter += 1
        self.replay.iter = self.__iter
        return action

    def reward(self, r):
        self.replay.rewards.append(r)
        self.total_reward += r

    def forward(self, x):
        raise NotImplementedError

    def on_reset(self, is_terminal):
        raise NotImplementedError

    def set_replay_hidden(self, hidden):
        self.replay.hidden0 = hidden

    def reset(self, new_episode=True):
        if new_episode:
            self.__iter = 0
            self.total_reward = 0.0
        self.replay = Replay()
        self.on_reset(new_episode)

    def end_replay(self, is_terminal):
        replay = self.replay
        replay.is_terminal = is_terminal
        replay.iter = self.__iter
        replay.total_reward = self.total_reward
        if not is_terminal:
            replay.rewards[-1] = self.last_value
            self.reset(new_episode=False)
        if replay.hidden0 is None:
            replay.hidden0 = torch.zeros(1)
        return replay
