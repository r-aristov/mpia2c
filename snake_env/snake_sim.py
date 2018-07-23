import numpy as np

SNAKE_HERBIVORE = 1
SNAKE_CARNIVORE = 2
SNAKE_OMNIVORE  = 3

OBJ_EMPTY   = 0
OBJ_WALL    = 1
OBJ_FOOD0   = 2
OBJ_FOOD1   = 3

DIR_LEFT =  0
DIR_UP =    1
DIR_RIGHT = 2
DIR_DOWN =  3

DIR_UPLEFT  =   4
DIR_UPRIGHT =   5
DIR_DOWNLEFT =  6
DIR_DOWNRIGHT = 7

class Snake:
    def __init__(self, x, y, snake_type=SNAKE_HERBIVORE, decision_interval=1, step_period=1, place=True):
        self.id = 0
        self.type = snake_type
        self.x = x
        self.y = y
        self.step_period = step_period
        self.decision_interval = decision_interval
        self.collected = {OBJ_FOOD0: 0, OBJ_FOOD1: 0}
        self.size = 3
        self.segments = []
        self.brain = None
        
        self.food = {OBJ_FOOD0: 1, OBJ_FOOD1: 1}
        self.hunger_iter = {OBJ_FOOD0: 1, OBJ_FOOD1: 1}

        # TODO: not the best way to place snake, should check for walls, etc...
        if place:
            self.segments.append((x+1,y+1))
            self.segments.append((x+1,y))
            self.segments.append((x,y))

    def clone(self):
        new_snake = Snake(self.x, self.y, self.type, self.decision_interval, False)
        new_snake.id = self.id
        new_snake.collected = self.collected.copy()
        new_snake.size = self.size
        new_snake.segments = self.segments.copy()
        new_snake.food = self.food.copy()
        new_snake.hunger_iter = self.hunger_iter
        return new_snake

    def put_on_map(self, world_map):
        map_id = 0xFF + self.id
        for seg in self.segments:
            x, y = seg
            world_map[x, y] = map_id

    def remove_from_map(self, world_map):
        for seg in self.segments:
            x, y = seg
            world_map[x, y] = 0


class SnakeSim:
    def __init__(self, width, height):
        self.ids = 0

        self.width = width
        self.height = height
        self.diagonal = np.sqrt(self.width**2 + self.height**2)

        self.map = np.zeros((width, height), dtype=np.uint16)
        self.iteration = 0

        self.on_successful_step = None
        self.on_wall_collide = None
        self.on_food_collected = None
        self.on_self_collide = None
        self.on_snake_collide = None
        self.on_need_decision = None

        self.on_step_begin = None
        self.on_step_end = None

        for x in range(0, width):
            self.map[x, 0] = 1
            self.map[x, self.height-1] = 1

        for y in range(0, height):
            self.map[0, y] = 1
            self.map[self.width-1, y] = 1

        self.objects = {}
        self.snakes = []


    def clone(self):
        new_sim = SnakeSim(self.width, self.height)
        new_sim.map = self.map.copy()
        new_sim.iteration = self.iteration
        new_sim.ids = self.ids

        new_sim.objects = {}
        for k, v in self.objects.items():
            new_sim.objects[k] = v.clone()

        new_sim.snakes = []
        for s in self.snakes:
            new_sim.snakes.append(new_sim.objects[s.id])

        return new_sim

    def process_snake(self, snake):
        hx, hy = snake.segments[-1]
        x1, y1 = snake.segments[-2]
        tx, ty = snake.segments[0]

        dx = hx - x1
        dy = hy - y1

        self.on_step_begin(self, snake)

        if self.iteration % snake.decision_interval == 0:
            direction = self.on_need_decision(self, snake, SnakeSim.delta_to_dir(dx, dy))
            if direction == DIR_LEFT and dx != 1:
                dx = -1
                dy = 0
            elif direction == DIR_RIGHT and dx != -1:
                dx = 1
                dy = 0
            elif direction == DIR_UP and dy != -1:
                dx = 0
                dy = 1
            elif direction == DIR_DOWN and dy != 1:
                dx = 0
                dy = -1

        nx = hx+dx
        ny = hy+dy

        if self.map[nx, ny] == OBJ_EMPTY:
            new_seg = (nx, ny)
            snake.segments.append(new_seg)
            self.map[nx, ny] = 0xFF+snake.id
            del snake.segments[0]
            self.map[tx, ty] = OBJ_EMPTY
            snake.x = nx
            snake.y = ny
            self.on_successful_step(self, snake, nx, ny)


        elif self.map[nx, ny] == OBJ_WALL:
            self.on_wall_collide(self, snake, nx, ny)

        elif self.map[nx, ny] == OBJ_FOOD0 or self.map[nx, ny] == OBJ_FOOD1:
            snake.x = nx
            snake.y = ny
            grow = self.on_food_collected(self, snake, nx, ny, self.map[nx, ny])
            new_seg = (nx, ny)
            if grow:
                snake.segments.append(new_seg)
                self.map[nx, ny] = 0xFF+snake.id
            else:
                snake.segments.append(new_seg)
                self.map[nx, ny] = 0xFF+snake.id
                del snake.segments[0]
                self.map[tx, ty] = OBJ_EMPTY

        elif self.map[nx, ny] > 0xFF:
            obj = self.objects[self.map[nx, ny]-0xFF]
            name = obj.__class__.__name__
            if name == "Snake":
                if obj.id == snake.id:
                    self.on_self_collide(self, snake, nx, ny)
                else:
                    self.on_snake_collide(self, snake, nx, ny, obj)

        self.on_step_end(self, snake)

    def step(self):
        for s in self.snakes:
            if self.iteration % s.step_period == 0:
                self.process_snake(s)
        self.iteration += 1

    def add_snake(self, snake):
        self.ids += 1
        snake.id = self.ids
        self.snakes.append(snake)
        self.objects[snake.id] = snake
        snake.put_on_map(self.map)

    @staticmethod
    def delta_to_dir(dx, dy):
        if dx == -1 and dy == 0:
            return DIR_LEFT
        if dx == 1 and dy == 0:
            return DIR_RIGHT
        if dx == 0 and dy == 1:
            return DIR_UP
        if dx == 0 and dy == -1:
            return DIR_DOWN

    @staticmethod
    def dir_to_delta(direction):
        dx = 0
        dy = 0

        if direction == DIR_LEFT:
            dx = -1
            dy = 0
        elif direction == DIR_RIGHT:
            dx = 1
            dy = 0
        elif direction == DIR_UP:
            dx = 0
            dy = 1
        elif direction == DIR_DOWN:
            dx = 0
            dy = -1
        elif direction == DIR_UPLEFT:
            dx = -1
            dy = 1
        elif direction == DIR_UPRIGHT:
            dx = 1
            dy = 1
        elif direction == DIR_DOWNRIGHT:
            dx = 1
            dy = -1
        elif direction == DIR_DOWNLEFT:
            dx = -1
            dy = -1
        return dx, dy

    @staticmethod
    def get_objects_on_line(sx, sy, ox, oy, k):
        # what Ys on line should be
        if k == 0:
            ly = np.full_like(ox, sy)

        elif np.isinf(k):
            lx = np.full_like(oy, sx)
            intersect_mask = lx == ox
            return ox[intersect_mask], oy[intersect_mask]

        else:
            b = sy - k*sx
            ly = k*ox + b
        # compare to existing Ys
        intersect_mask = ly == oy
        # objects on line
        return ox[intersect_mask], oy[intersect_mask]

    @staticmethod
    def get_line_distance(ox, oy, sx, sy, vertical=False):
        d = np.sqrt((ox-sx)**2 + (oy-sy)**2)

        if vertical:
            mask = (oy < sy)
        else:
            mask = (ox < sx)

        inv_mask = ~mask
        ld = d[mask]
        rd = d[inv_mask]

        if ld.size == 0:
            ld = np.array([-13.0])
        if rd.size == 0:
            rd = np.array([-13.0])
        return np.min(ld), np.min(rd)

    @staticmethod
    def get_vector(map_array, obj_type, sx, sy, add_zero=False):
        ox, oy = np.where(map_array == obj_type)
        ox_diag_main, oy_diag_main = SnakeSim.get_objects_on_line(sx, sy, ox, oy, 1)
        ox_diag_sec, oy_diag_sec = SnakeSim.get_objects_on_line(sx, sy, ox, oy, -1)
        ox_horizontal, oy_horizontal = SnakeSim.get_objects_on_line(sx, sy, ox, oy, 0)
        ox_vertical, oy_vertical = SnakeSim.get_objects_on_line(sx, sy, ox, oy, np.inf)

        d_ul, d_dr = SnakeSim.get_line_distance(ox_diag_main, oy_diag_main, sx, sy)
        d_dl, d_ur = SnakeSim.get_line_distance(ox_diag_sec, oy_diag_sec, sx, sy)
        d_l, d_r = SnakeSim.get_line_distance(ox_horizontal, oy_horizontal, sx, sy)
        d_u, d_d = SnakeSim.get_line_distance(ox_vertical, oy_vertical, sx, sy, vertical=True)

        if add_zero:
            return [d_ul, d_u, d_ur, d_l, 0, d_r, d_dl, d_d, d_dr]
        return [d_ul, d_u, d_ur, d_l, d_r, d_dl, d_d, d_dr]
