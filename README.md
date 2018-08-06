mpia2c
=============

**mpia2c** is a distributed python implementation of Advantage-Actor-Critic reinforcment learning algorithm which uses mpi4py library to run 
agents on multiple nodes.

Dependencies:
 - pytorch (https://pytorch.org/)
 - mpi4py	(http://mpi4py.scipy.org/docs/)
 - numpy	(http://www.numpy.org/)
 
 To run examples you need Open AI Gym (https://gym.openai.com/) for "cartpole" example and pygame (https://www.pygame.org/) for "snake"

Examples:
-------
To run **cartpole** example:
```
mpiexec -n 11 python cartpole_train.py trained\cartpole.nn --gamma=0.99
```

To run **snake** example (training):
```
mpiexec -n 11 python snake_train.py trained\saved_network.nn --gamma=0.97
```

```
usage: cartpole_train.py [-h] [--src SRC] [--gamma GAMMA] [--lr LR]
                         [--log-interval LOG_INTERVAL]
                         [--save-interval SAVE_INTERVAL]
                         [--ppo-iters PPO_ITERS] [--ppo-clip PPO_CLIP]
                         [--steps-in-replay STEPS_IN_REPLAY]
                         [--iterations ITERATIONS] [--seed SEED]
                         dst
                      
positional arguments:
  dst                   trained model save path

optional arguments:
  -h, --help            show this help message and exit
  --src SRC             saved model to resume training
  --gamma GAMMA         discount factor (default: 0.99)
  --lr LR               discount factor (default: 3e-3)
  --log-interval LOG_INTERVAL
                        interval between training status logs [in full
                        episodes] (default: 10)
  --save-interval SAVE_INTERVAL
                        interval between saving model weights [in full
                        episodes] (default: 10)
  --ppo-iters PPO_ITERS
                        number ppo iterations. If value is 1, vanilla a2c is
                        used (default: 4)
  --ppo-clip PPO_CLIP   ppo loss clipping value (default: 0.2)
  --steps-in-replay STEPS_IN_REPLAY
                        max steps in replay (default: 500)
  --iterations ITERATIONS
                        iterations to train (default: 1e8)
  --seed SEED           random seed
```   
Note: trainig  scripts should be started via MPI (follow https://mpi4py.scipy.org/docs/usrman/tutorial.html and MPI manuals to learn more about starting parameters of mpiexec) 

To run **snake** example (test):
```
python snake_test trained\snake.nn
```
[![snake trained](https://github.com/r-aristov/mpia2c/blob/master/snake.gif)]
