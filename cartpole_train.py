import argparse
import gym
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from mpi_a2c import MPIA2C, A2CAgent
from mpi4py import MPI
import time


class GymEnvironment:
    def __init__(self, agent, rank):
        self.done = False
        
        self.agent = agent
        self.rank = rank
        self.env = gym.make('CartPole-v0')
        self.last_state = self.env.reset()
        self.sim, self.controller, self.snake = (None, None, None)

    def reset(self):
        self.done = False
        self.last_state = self.env.reset()

    def step(self):
        action = self.agent.decide(torch.tensor(self.last_state).float())
        new_state, reward, done, _ = self.env.step(action)
        self.last_state = new_state
        self.agent.reward(reward)
        self.done = done

        
class Brain(nn.Module, A2CAgent):
    def __init__(self):
        nn.Module.__init__(self)
        A2CAgent.__init__(self)

        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        value = self.value_head(x)
        return F.softmax(action_scores, dim=-1), value

    def save(self, filename):
        f = open(filename, "wb")
        torch.save(self.state_dict(), f)
        f.close()

    def load(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)

    def on_reset(self, new_episode):
        pass


class ProgressTracker:
    def __init__(self, agent, threshold):
        self.agent = agent
        self.running_reward = 1.0
        self.running_iter = 1.0
        self.lst_tick = time.time()
        self.episode = 0
        self.total_time = 0.0
        self.episode_reward = 0.0
        self.threshold = threshold

    def on_batch_done(self, loss, batch):
        if batch % 20 == 0:
            dt = time.time() - self.lst_tick
            self.lst_tick = time.time()
            self.total_time += dt

            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(self.episode,  round(self.episode_reward), self.running_reward))
            if self.running_reward > 195:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(self.running_reward, round(self.episode_reward)))
                sys.exit(0)

    def on_episodes_done(self, episode, batch_rewards, batch_iters):
        total_reward_avg = torch.tensor(batch_rewards).float().mean().item()
        self.episode_reward = total_reward_avg
        iter_avg = torch.tensor(batch_iters).float().mean().item()
        self.running_reward = self.running_reward * 0.99 + total_reward_avg * 0.01
        self.running_iter = self.running_iter * 0.99 + iter_avg * 0.01
        self.episode = episode
        if episode % 10 == 0:
            self.agent.save("trained\\brain_a2c.nn")        


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    MPIA2C.init_mpi_rng()

    if size == 1:
        print("Can not run on single node, 2 nodes required at least!")
        sys.exit(-1)

    parser = argparse.ArgumentParser(description='MPI A2C')

    parser.add_argument('dst', type=str,
                        help='trained model save path')

    parser.add_argument('--src', type=str, default='',
                        help='saved model to resume training')

    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor (default: 0.99)')

    parser.add_argument('--lr', type=float, default=3e-3,
                        help='discount factor (default: 3e-3)')

    parser.add_argument('--log-interval', type=int, default=20,
                        help='interval between training status logs [in full replays] (default: 20)')

    parser.add_argument('--save-interval', type=int, default=10,
                        help='interval between saving model weights [in full episodes] (default: 10)')

    parser.add_argument('--replays-in-batch', type=int, default=5,
                        help='number of replays in batch (per estimator) (default: 5)')

    parser.add_argument('--steps-in-replay', type=int, default=500,
                        help='max steps in replay (default: 500)')

    parser.add_argument('--iterations', type=int, default=1e8,
                        help='iterations to train (default: 1e8)')
    args = parser.parse_args()

    if rank == 0:
        print("Starting RL training on %d nodes..." % size)
        print("Model weights will be saved to %s every %d full episodes" % (args.dst, args.save_interval))
        print("Gamma = %1.3f, LR = %1.4f" % (args.gamma, args.lr))

    agent = Brain()
    if args.src != "":
        agent.load(args.src)
        if rank == 0:
            print("Weights loaded from %s" % args.src)

    gym_env = GymEnvironment(agent, rank)
    ptracker = ProgressTracker(agent, gym_env.env.spec.reward_threshold)
    ptracker.log_interval = args.log_interval
    ptracker.save_interval = args.save_interval
    ptracker.dst = args.dst
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    a2c = MPIA2C(comm, rank, optimizer, on_batch_done=ptracker.on_batch_done, on_episodes_done=ptracker.on_episodes_done)
    a2c.max_train_iters = args.iterations
    a2c.steps_in_replay = args.steps_in_replay
    a2c.gamma = args.gamma
    a2c.replays_in_batch = args.replays_in_batch

    a2c.agent = agent
    a2c.env = gym_env
    a2c.run()


if __name__ == '__main__':
    main()
