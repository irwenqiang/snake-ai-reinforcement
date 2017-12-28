import numpy as np
import pygame
import time

from snakeai.agent import HumanAgent
from snakeai.gameplay.entities import (CellType, SnakeAction, ALL_SNAKE_DIRECTIONS)


class PyGameGUI:
    """ Provides a Snake GUI powered by Pygame. """

    FPS_LIMIT = 60
    AI_TIMESTEP_DELAY = 200
    HUMAN_TIMESTEP_DELAY = 200
    CELL_SIZE = 20

    SNAKE_CONTROL_KEYS = [
        pygame.K_UP,
        pygame.K_LEFT,
        pygame.K_DOWN,
        pygame.K_RIGHT
    ]

    def __init__(self):
        pygame.init()
        self.agents = [HumanAgent(), HumanAgent()]
        self.env = None
        self.screen = None
        self.fps_clock = None
        self.timestep_watch = Stopwatch()

    def load_environment(self, environment):
        """ Load the RL environment into the GUI. """
        self.env = environment
        screen_size = (self.env.field.size * self.CELL_SIZE, self.env.field.size * self.CELL_SIZE)
        self.screen = pygame.display.set_mode(screen_size)
        self.screen.fill(Colors.SCREEN_BACKGROUND)
        pygame.display.set_caption('Snake')

    def load_agent(self, agents):
        """ Load the RL agent into the GUI. """
        self.agents = agents

    def render_cell(self, x, y):
        """ Draw the cell specified by the field coordinates. """
        cell_coords = pygame.Rect(
            x * self.CELL_SIZE,
            y * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE,
        )
        if self.env.field[x, y] == CellType.EMPTY:
            pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
        else:
            color = Colors.CELL_TYPE[self.env.field[x, y]]
            pygame.draw.rect(self.screen, color, cell_coords, 1)

            internal_padding = self.CELL_SIZE // 6 * 2
            internal_square_coords = cell_coords.inflate((-internal_padding, -internal_padding))
            pygame.draw.rect(self.screen, color, internal_square_coords)

    def render(self):
        """ Draw the entire game frame. """
        for x in range(self.env.field.size):
            for y in range(self.env.field.size):
                self.render_cell(x, y)

    def map_key_to_snake_action(self, key):
        """ Convert a keystroke to an environment action. """
        actions = [
            SnakeAction.MAINTAIN_DIRECTION,
            SnakeAction.TURN_LEFT,
            SnakeAction.MAINTAIN_DIRECTION,
            SnakeAction.TURN_RIGHT,
        ]

        key_idx = self.SNAKE_CONTROL_KEYS.index(key)
        direction_idx = ALL_SNAKE_DIRECTIONS.index(self.env.snake.direction)
        return np.roll(actions, -key_idx)[direction_idx]

    def run(self, num_episodes=1):
        """ Run the GUI player for the specified number of episodes. """
        pygame.display.update()
        self.fps_clock = pygame.time.Clock()

        try:
            for episode in range(num_episodes):
                self.run_episode()
                pygame.time.wait(1500)
        except QuitRequestedError:
            pass

    def run_episode(self):
        """ Run the GUI player for a single episode. """

        # Initialize the environment.
        self.timestep_watch.reset()
        timestep_result = self.env.new_episode()
        for i in range(2):
            self.agents[i].begin_episode()

        is_human_agent = isinstance(self.agents[0], HumanAgent)
        timestep_delay = self.HUMAN_TIMESTEP_DELAY if is_human_agent else self.AI_TIMESTEP_DELAY

        # Main game loop.
        running = True
        while running:
            actions = [SnakeAction.MAINTAIN_DIRECTION for i in range(2)]

            # Handle events.
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if is_human_agent and event.key in self.SNAKE_CONTROL_KEYS:
                        for i in range(2):
                            actions[i] = self.map_key_to_snake_action(event.key)
                    if event.key == pygame.K_ESCAPE:
                        raise QuitRequestedError

                if event.type == pygame.QUIT:
                    raise QuitRequestedError

            # Update game state.
            timestep_timed_out = self.timestep_watch.time() >= timestep_delay
            human_made_move = is_human_agent and action != SnakeAction.MAINTAIN_DIRECTION

            if timestep_timed_out or human_made_move:
                self.timestep_watch.reset()

                if not is_human_agent:
                    for i in range(2):
                        if i == 0:
                            print("observation:")
                            print(np.where(timestep_result[i].observation==21))
                            print(timestep_result[i].observation)
                            print(timestep_result[i].observation[np.where(timestep_result[i].observation==21)])
			    # if agent 0 then agent's head is 2 and agent1 is ignore
			    # 20 agent 0
			    # 21 agent 1
                            timestep_result[i].observation[np.where(timestep_result[i].observation==21)]=0
                            timestep_result[i].observation[np.where(timestep_result[i].observation==20)]=2
                            timestep_result[i].observation[np.where(timestep_result[i].observation==31)]=0
                            timestep_result[i].observation[np.where(timestep_result[i].observation==30)]=3
                            print(timestep_result[i].observation)
                            #timestep_result[i].observation
                        elif i == 1:
                            timestep_result[i].observation[np.where(timestep_result[i].observation==21)]=2
                            timestep_result[i].observation[np.where(timestep_result[i].observation==20)]=0
                            timestep_result[i].observation[np.where(timestep_result[i].observation==30)]=0
                            timestep_result[i].observation[np.where(timestep_result[i].observation==31)]=3
                        #actions[i] = self.agents[i].act(timestep_result[i].observation, timestep_result[i].reward)	
                        actions[i] = self.agents[i].act(timestep_result[i].observation, None)	
                        print(actions)

                self.env.choose_action(actions)
                timestep_result = self.env.timestep()

                for i in range(2):
                    if timestep_result[i].is_episode_end:
                        self.agents[i].end_episode()
                        running = False

                #if timestep_result[0].is_episode_end and timestep_result[1].is_episode_end:
                #    self.agents[0].end_episode()
                #    self.agents[1].end_episode()
                #    running = False
            # Render.
            self.render()
            score0 = self.env.snakes[0].length - self.env.initial_snake_length
            score1 = self.env.snakes[1].length - self.env.initial_snake_length
            pygame.display.set_caption(f'[Score: {score0:02d}-{score1:02d}]')
            pygame.display.update()
            self.fps_clock.tick(self.FPS_LIMIT)


class Stopwatch(object):
    """ Measures the time elapsed since the last checkpoint. """

    def __init__(self):
        self.start_time = pygame.time.get_ticks()

    def reset(self):
        """ Set a new checkpoint. """
        self.start_time = pygame.time.get_ticks()

    def time(self):
        """ Get time (in milliseconds) since the last checkpoint. """
        return pygame.time.get_ticks() - self.start_time


class Colors:

    SCREEN_BACKGROUND = (170, 204, 153)
    CELL_TYPE = {
        CellType.WALL: (56, 56, 56),
        #CellType.SNAKE_BODY0: (105, 132, 164),
        #CellType.SNAKE_BODY1: (164, 132, 105),
        CellType.SNAKE_HEAD0: (0, 120, 0),
        CellType.SNAKE_HEAD1: (0, 0, 120),
        CellType.SNAKE_BODY0: (0, 255, 0),
        CellType.SNAKE_BODY1: (0, 0, 255),
        CellType.FRUIT: (173, 52, 80),
    }


class QuitRequestedError(RuntimeError):
    """ Gets raised whenever the user wants to quit the game. """
    pass
