import time

import numpy as np
import torch
from torch import nn as nn, optim as optim
from torch.nn import functional as F

from mpi_a2c import MPIA2C, A2CAgent
from snake_env.snake_sim import SnakeSim, Snake, OBJ_WALL, OBJ_FOOD0, OBJ_FOOD1, OBJ_EMPTY
from mpi4py import MPI
import argparse
import sys


class SnakeEnvironment:
    def __init__(self, agent, rank):
        self.done = False
        self.agent = agent
        self.rank = rank
        self.sim, self.controller, self.snake = (None, None, None)

    def reset(self):
        self.done = False
        self.sim, self.controller, self.snake = sim_init(self.agent)

    def step(self):
        self.sim.step()
        self.done = self.controller.died
        r = self.controller.current_r
        self.agent.reward(r)


class Brain(nn.Module, A2CAgent):
    def __init__(self):
        nn.Module.__init__(self)
        A2CAgent.__init__(self)

        self.state_features_count = 3+4*8
        self.linear = nn.Linear(self.state_features_count, 64)
        self.gru = nn.GRUCell(64, 64)
        self.action_head = nn.Linear(64, 4)
        self.value_head = nn.Linear(64, 1)
        self.hidden = torch.zeros(1, 64)

        self.sequence = 0
        self.seq_len = 500
        self.act0 = None
        self.act1 = None

    def forward(self, x):
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        input_features = F.relu(self.linear(x))
        self.act0 = input_features.squeeze()
        self.hidden = self.gru(input_features, self.hidden)
        self.act1 = self.hidden.squeeze()
        action_scores = self.action_head(self.hidden)
        value = self.value_head(self.hidden)

        self.sequence += 1
        if self.sequence >= self.seq_len:
            self.hidden.detach_()
            self.sequence = 0

        return F.softmax(action_scores, dim=-1), value

    def save(self, filename):
        f = open(filename, "wb")
        torch.save(self.state_dict(), f)
        f.close()

    def load(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)

    def on_reset(self, new_episode):
        self.sequence = 0
        if new_episode:
            self.hidden = torch.zeros(1, 64)
        self.hidden.detach_()
        self.set_replay_hidden(self.hidden)


class ProgressTracker:
    def __init__(self, agent):
        self.agent = agent
        self.running_reward = 1.0
        self.running_iter = 1.0
        self.lst_tick = time.time()
        self.episode = 0
        self.log_interval = 20
        self.save_interval = 10
        self.total_time = 0.0
        self.dst = ''

    def on_batch_done(self, loss, batch):
        pass


    def on_episodes_done(self, episode, batch_rewards, batch_iters):
        total_reward_avg = torch.tensor(batch_rewards).float().mean().item()
        iter_avg = torch.tensor(batch_iters).float().mean().item()
        self.running_reward = self.running_reward * 0.9 + total_reward_avg * 0.1
        self.running_iter = self.running_iter * 0.9 + iter_avg * 0.1
        self.episode = episode

        if episode % self.log_interval == 0 and episode != 0:
            dt = time.time() - self.lst_tick
            self.lst_tick = time.time()
            self.total_time += dt
            log_str = "{:<15.2f}{:<15}{:<15.2f}{:<15.2f}".format(self.total_time, self.episode, self.running_iter, self.running_reward)
            print(log_str)

        if episode % self.save_interval == 0:
            self.agent.save(self.dst)


class SnakeEnvController:
    def __init__(self):
        self.food_available = {OBJ_FOOD0: 0.0, OBJ_FOOD1: 0.0}

        self.died = False
        self.food_nutrition = {OBJ_FOOD0: 1.0, OBJ_FOOD1: 1.0}

        self.wall_collisions = 0

        self.current_r = 0.0
        self.total_reward = 0.0

        self.collected_reward = {OBJ_FOOD0: 0.0, OBJ_FOOD1: 0.0}
        self.starvation_reward = {OBJ_FOOD0: -1.0, OBJ_FOOD1: -1.0}

        self.step_reward = 0.0
        self.wall_collision_reward = -1.0
        self.self_collision_reward = -1.0

        self.grow_limit = 3
        self.die_on_starvation = {OBJ_FOOD0: True, OBJ_FOOD1: True}
        self.die_on_collision = True

        self.clustered = {OBJ_FOOD0: False, OBJ_FOOD1: False}
        self.cluster_size = {OBJ_FOOD0: 15.0, OBJ_FOOD1: 15.0}
        self.food_limit = {OBJ_FOOD0: 100.0, OBJ_FOOD1: 100.0}
        self.got_food = {OBJ_FOOD0: 0.0, OBJ_FOOD1: 0.0}

    def reset(self):
        self.died = False
        self.wall_collisions = 0
        self.current_r = 0.0
        self.total_reward = 0.0

    def successful_step(self, sim, snake, nx, ny):
        for k in snake.food:
            if sim.iteration % snake.hunger_iter[k] == 0:
                snake.food[k] -= 1

            if snake.food[k] < 0:
                self.died = self.die_on_starvation[k]
                self.current_r += self.starvation_reward[k]

        self.current_r += self.step_reward

    def wall_collide(self, sim, snake, nx, ny):
        self.died = self.die_on_collision
        self.wall_collisions += 1
        self.current_r += self.wall_collision_reward

    def food_collected(self, sim, snake, nx, ny, food_type):
        self.got_food[food_type] = 1.0
        snake.collected[food_type] += 1
        snake.food[food_type] += self.food_nutrition[food_type]

        if snake.food[food_type] > self.food_limit[food_type]:
            snake.food[food_type] = self.food_limit[food_type]
        self.current_r += self.collected_reward[food_type]
        self.food_available[food_type] -= 1

        if self.clustered[food_type]:
            if self.food_available[food_type] == 0:
                self.gen_clusters(sim, 1, self.cluster_size[food_type], 1.5e-2, food_type)
                self.food_available[food_type] += self.cluster_size[food_type]
        else:
            self.gen_obj(sim, 1, food_type)

        if snake.collected[OBJ_FOOD0] < self.grow_limit - 3:
            return True

        return False

    def self_collide(self, sim, snake, nx, ny):
        self.died = self.die_on_collision
        self.current_r += self.self_collision_reward

    def snake_collide(self, sim, snake, nx, ny, other):
        pass

    def need_decision(self, sim, snake, direction):
        k = 10.0 / sim.diagonal

        v_wall = SnakeSim.get_vector(sim.map, OBJ_WALL, snake.x, snake.y)
        v_food0 = SnakeSim.get_vector(sim.map, OBJ_FOOD0, snake.x, snake.y)
        v_food1 = SnakeSim.get_vector(sim.map, OBJ_FOOD1, snake.x, snake.y)

        # to avoid zero distances
        t = sim.map[snake.x, snake.y]
        sim.map[snake.x, snake.y] = -1
        v_self = SnakeSim.get_vector(sim.map, 0xFF + snake.id, snake.x, snake.y)
        sim.map[snake.x, snake.y] = t

        v_food0 = torch.tensor(v_food0) * k
        v_food1 = torch.tensor(v_food1) * k
        v_wall = torch.tensor(v_wall) * k
        v_self = torch.tensor(v_self) * k

        f = snake.food[OBJ_FOOD0]
        w = snake.food[OBJ_FOOD1]

        hunger = torch.tensor([f / self.food_limit[OBJ_FOOD0]])
        thirst = torch.tensor([w / self.food_limit[OBJ_FOOD1]])

        reserved = torch.zeros(1)

        v = torch.cat((v_food0, v_food1, v_wall, v_self, hunger, thirst, reserved))
        action = snake.brain.decide(v)
        return action

    def step_begin(self, sim, snake):
        self.current_r = 0

    def step_end(self, sim, snake):
        self.total_reward += self.current_r

    def gen_obj(self, sim, n, type):
        size = sim.map.size
        mask = (sim.map == OBJ_EMPTY).astype(np.float32)
        probs = mask/mask.sum()
        probs = probs.flatten()
        places = np.random.choice(size, n, p=probs, replace=False)
        sim.map.flat[places] = type

    def gen_clusters(self, sim, clusters, n, d, type):
        w = sim.width
        h = sim.height
        size = sim.map.size

        mask = sim.map == OBJ_EMPTY
        ox, oy = np.where(mask)
        idx = np.random.choice(ox.size, clusters, replace=False)
        x = np.arange(0, w).reshape((w, 1))
        y = np.arange(0, h).reshape((1, h))

        sx = ox[idx].reshape((-1, clusters))
        sy = oy[idx].reshape((clusters, 1))

        fx = np.transpose(np.reshape((x-sx) ** 2, (w, clusters, 1)))
        fx = np.broadcast_to(fx, (h, clusters, w))

        fy = np.transpose(np.reshape((y-sy) ** 2, (1, clusters, h)))
        fy = np.broadcast_to(fy, (w, clusters, h))

        distances = fx+fy
        distances = np.prod(distances, 1)
        md = np.max(distances)
        distances[distances > d*md] = md
        distances = 1.0/(distances + 1e-10)

        distances *= mask

        probs = distances/np.sum(distances)
        probs = probs.flatten()
        places = np.random.choice(size, n, p=probs, replace=False)
        sim.map.flat[places] = type


def gen_cluster_or_sparse(sim, controller, count, food_type, clusters=1, cluster_density=9e-3):
    if controller.clustered[food_type]:
        controller.gen_clusters(sim, 1, count, cluster_density, food_type)
    else:
        controller.gen_obj(sim, count, food_type)


def sim_init(brain):
    w, h = (40, 40)
    sim = SnakeSim(w, h)
    controller = SnakeEnvController()

    # callbacks
    sim.on_successful_step = controller.successful_step
    sim.on_wall_collide = controller.wall_collide
    sim.on_food_collected = controller.food_collected
    sim.on_self_collide = controller.self_collide
    sim.on_snake_collide = controller.snake_collide
    sim.on_need_decision = controller.need_decision

    sim.on_step_begin = controller.step_begin
    sim.on_step_end = controller.step_end

    # sim init
    x = np.random.random_integers(4, sim.width - 4)
    y = np.random.random_integers(4, sim.height - 4)

    snake = Snake(x, y, decision_interval=1)
    snake.brain = brain
    sim.add_snake(snake)

    food = np.random.randint(10, 30)
    water = np.random.randint(10, 30)

    controller.clustered[OBJ_FOOD0] = False
    controller.clustered[OBJ_FOOD1] = True
    controller.cluster_size[OBJ_FOOD0] = int(food)
    controller.cluster_size[OBJ_FOOD1] = int(water)
    controller.food_limit[OBJ_FOOD0] = 200.0
    controller.food_limit[OBJ_FOOD1] = 200.0
    snake.food[OBJ_FOOD0] = np.random.randint(10, 200)
    snake.food[OBJ_FOOD1] = np.random.randint(10, 200)

    gen_cluster_or_sparse(sim, controller, food, OBJ_FOOD0)
    gen_cluster_or_sparse(sim, controller, water, OBJ_FOOD1)

    controller.food_available[OBJ_FOOD0] = food
    controller.food_available[OBJ_FOOD1] = water

    controller.food_nutrition[OBJ_FOOD0] = np.random.randint(10, 40)
    controller.food_nutrition[OBJ_FOOD1] = np.random.randint(10, 40)
    snake.hunger_iter = {OBJ_FOOD0: 1, OBJ_FOOD1: 1}
    controller.grow_limit = 3

    # rewards & death conditions
    controller.collected_reward = {OBJ_FOOD0: 0.0, OBJ_FOOD1: 0.0}
    controller.step_reward = 0.00
    controller.starvation_reward = {OBJ_FOOD0: -2.0, OBJ_FOOD1: -2.0}
    controller.wall_collision_reward = -2.0
    controller.self_collision_reward = -2.0
    controller.die_on_starvation = {OBJ_FOOD0: True, OBJ_FOOD1: True}
    controller.die_on_collision = True

    return sim, controller, snake


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    MPIA2C.init_mpi_rng(0xAABBFEFE)

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

    parser.add_argument('--log-interval', type=int, default=10,
                        help='interval between training status logs [in full episodes] (default: 10)')

    parser.add_argument('--save-interval', type=int, default=10,
                        help='interval between saving model weights [in full episodes] (default: 10)')

    parser.add_argument('--ppo-iters', type=int, default=4,
                        help='number ppo iterations. If value is 1, vanilla a2c is used (default: 4)')

    parser.add_argument('--ppo-clip', type=float, default=0.2,
                        help='ppo loss clipping value (default: 0.2)')

    parser.add_argument('--steps-in-replay', type=int, default=500,
                        help='max steps in replay (default: 500)')

    parser.add_argument('--iterations', type=int, default=1e8,
                        help='iterations to train (default: 1e8)')
    args = parser.parse_args()

    args.ppo_iters = args.ppo_iters if args.ppo_iters > 0 else 1

    if rank == 0:
        print("Starting RL training on %d nodes..." % size)
        print("Model weights will be saved to %s every %d full episodes" % (args.dst, args.save_interval))
        print("Gamma = %1.3f, LR = %1.4f" % (args.gamma, args.lr))
        if args.ppo_iters == 1:
            print("Using vanilla A2C loss (ppo_iters = 1)")
        else:
            print("Using PPO loss (ppo iters = %d, ppo clip = %1.2f)" % (args.ppo_iters, args.ppo_clip))

    agent = Brain()
    if args.src != "":
        agent.load(args.src)
        if rank == 0:
            print("Weights loaded from %s" % args.src)

    bmgr = ProgressTracker(agent)
    bmgr.log_interval = args.log_interval
    bmgr.save_interval = args.save_interval
    bmgr.dst = args.dst
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    a2c = MPIA2C(comm, rank, optimizer, on_batch_done=bmgr.on_batch_done, on_episodes_done=bmgr.on_episodes_done)
    a2c.max_train_iters = args.iterations
    a2c.steps_in_replay = args.steps_in_replay
    a2c.gamma = args.gamma
    a2c.ppo_iters = args.ppo_iters
    a2c.ppo_clip = args.ppo_clip

    a2c.agent = agent
    a2c.env = SnakeEnvironment(a2c.agent, rank)
    a2c.run()


if __name__ == '__main__':
    main()
