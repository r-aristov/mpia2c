import pygame
import sys
import time
import numpy as np

from pygame.locals import *
from snake_env import snake_sim
from snake_train import sim_init
from snake_train import Brain
from snake_env.snake_sim import OBJ_FOOD0, OBJ_FOOD1
import matplotlib.colors
import matplotlib.pyplot as plt
import torch

FPS = 60
WINDOWWIDTH = 800
WINDOWHEIGHT = 400

#             R    G    B
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

RED = (255, 0, 0)
DARKRED = (155, 0, 0)

GREEN = (0, 255, 0)
DARKGREEN = (0, 155, 0)

BLUE = (0, 0, 255)
DARKBLUE = (0, 0, 155)

DARKGRAY = (55, 55, 55)
GRAY = (155, 155, 155)

CYAN = (0, 200, 255)

BGCOLOR = BLACK

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

GRID_SIZE_X = 10
GRID_SIZE_Y = 10

cmap = plt.cm.hot
cmap_inv = plt.cm.hot_r

norm0 = matplotlib.colors.Normalize(vmin=0.0, vmax=5.0)
norm1 = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
norm_val = matplotlib.colors.Normalize(vmin=-1.0, vmax=0.0)

def main():
    global FPSCLOCK, DISPLAYSURF, BASICFONT

    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
    pygame.display.set_caption('Sim')

    while True:
        simulate_and_render()


def double_rect(x, y, w, h, c1, c2):
    pygame.draw.rect(DISPLAYSURF, c1, [x + 2, y + 2, w - 2, h - 2])
    pygame.draw.rect(DISPLAYSURF, c2, [x, y, w, h], 2)


def process_complex_object(sim, i, j, obj):
    x = i * GRID_SIZE_X
    y = j * GRID_SIZE_Y
    name = obj.__class__.__name__
    c0 = WHITE
    c1 = WHITE

    if name == "Snake":
        if obj.type == snake_sim.SNAKE_HERBIVORE:
            c0 = BLUE
            c1 = DARKBLUE
        elif obj.type == snake_sim.SNAKE_CARNIVORE:
            c0 = RED
            c1 = DARKRED

        double_rect(x, y, GRID_SIZE_X, GRID_SIZE_Y, c0, c1)


def render_map_and_objects(sim):
    for j in range(sim.height):
        for i in range(sim.width):
            obj_type_or_id = sim.map[i, j]
            x = i * GRID_SIZE_X
            y = j * GRID_SIZE_Y

            if obj_type_or_id > 0xFF:
                obj = sim.objects[obj_type_or_id - 0xFF]
                process_complex_object(sim, i, j, obj)

            elif obj_type_or_id == snake_sim.OBJ_WALL:
                double_rect(x, y, GRID_SIZE_X, GRID_SIZE_Y, GRAY, DARKGRAY)
            elif obj_type_or_id == snake_sim.OBJ_FOOD0:
                double_rect(x, y, GRID_SIZE_X, GRID_SIZE_Y, GREEN, DARKGREEN)
            elif obj_type_or_id == snake_sim.OBJ_FOOD1:
                double_rect(x, y, GRID_SIZE_X, GRID_SIZE_Y, CYAN, CYAN)


def init_all():
    b = Brain()
    if len(sys.argv) == 2:
        print("Loading network parameters from %s" % sys.argv[1])
        b.load(sys.argv[1])
    b.eval()
    b.reset()


    for p in b.parameters():
        p.detach_()

    sim, controller, snake = sim_init(b)
    return sim, snake, controller





def render_activations(act0, act1):
    i = 0
    j = 0
    act_points0 = cmap(norm0(act0))
    act_points1 = cmap(norm1(act1))
    for a0, a1 in zip(act_points0, act_points1):
        clr0 = a0[:3]*255.0
        clr1 = a1[:3]*255.0
        double_rect(430 + 20*i, 120+20*j, 10, 10, clr0.tolist(), clr0.tolist())
        double_rect(620 + 20*i, 120+20*j, 10, 10, clr1.tolist(), clr1.tolist())
        i += 1
        if i % 8 == 0 and i > 0:
            j += 1
            i = 0


def render_info(font, old_i, old_c, old_f, old_w, val):
    line0 = "steps: %d" % old_i
    line1 = "collisions: %d" % old_c
    line2 = "food: %d" % old_f
    line3 = "water: %d" % old_w
    line4 = "V: %1.4f" % val
    lines = [line0 + "        " + line1, line2 + "        " + line3, line4]
    i = 0
    for l in lines:
        textsurface = font.render(l, False, WHITE)
        DISPLAYSURF.blit(textsurface, (490, 5+i*25))
        i += 1


def simulate_and_render():
    DISPLAYSURF.fill(BGCOLOR)
    pygame.font.init()
    font = pygame.font.SysFont('Comic Sans MS', 20)

    tick_period = 0.015

    sim, snake, controller = init_all()

    precise_time = time.time()

    #just a way to display net desisions before the actually happen in sim - one step lag in rendering

    old_sim = sim.clone()
    old_c = controller.wall_collisions

    sim.step()

    act0 = snake.brain.act0
    act1 = snake.brain.act1
    val = snake.brain.last_value

    render_map_and_objects(old_sim)
    render_activations(act0.tolist(), act1.tolist())
    render_info(font, old_sim.iteration, old_c, old_sim.snakes[0].food[OBJ_FOOD0], old_sim.snakes[0].food[OBJ_FOOD1], val)

    val_clr = cmap_inv(norm_val([val]))[:3]*255.0
    double_rect(600, 65, 10, 10, val_clr[0], val_clr[0])

    pygame.display.update()
    pause = True

    while pause:
        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()
            elif event.type == KEYDOWN:
                if event.key == K_RETURN:
                    pause = False
                    break

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()
            elif event.type == KEYDOWN:
                if event.key == K_KP_PLUS:
                    tick_period /= 2
                elif event.key == K_KP_MINUS:
                    tick_period *= 2
                elif event.key == K_RETURN:
                    sim, snake, controller = init_all()
                elif event.key == K_ESCAPE:
                    terminate()

        DISPLAYSURF.fill(BGCOLOR)

        dt = time.time() - precise_time
        if dt > tick_period:
            old_sim = sim.clone()
            old_c = controller.wall_collisions

            sim.step()

            act0 = snake.brain.act0
            act1 = snake.brain.act1
            val = snake.brain.last_value
            precise_time = time.time()

        render_map_and_objects(old_sim)
        render_activations(act0.tolist(), act1.tolist())
        render_info(font, old_sim.iteration, old_c, old_sim.snakes[0].food[OBJ_FOOD0], old_sim.snakes[0].food[OBJ_FOOD1], val)
        val_clr = cmap_inv(norm_val([val]))[:3]*255.0
        double_rect(600, 65, 10, 10, val_clr[0], val_clr[0])
        pygame.display.update()

        FPSCLOCK.tick(FPS)


def terminate():
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
