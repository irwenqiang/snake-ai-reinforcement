import pprint
import random
import time

import numpy as np
import pandas as pd
import collections

from .entities import Snake, Field, CellType, SnakeAction, ALL_SNAKE_ACTIONS


class Environment(object):
    """
    Represents the RL environment for the Snake game that implements the game logic,
    provides rewards for the agent and keeps track of game statistics.
    """

    def __init__(self, config, verbose=1):
        """
        Create a new Snake RL environment.
        
        Args:
            config (dict): level configuration, typically found in JSON configs.  
            verbose (int): verbosity level:
                0 = do not write any debug information;
                1 = write a CSV file containing the statistics for every episode;
                2 = same as 1, but also write a full log file containing the state of each timestep.
        """
        #self.fields = [Field(level_map=config['field0']), Field(level_map=config['field1'])]
        self.field = Field(level_map=config['field'])
        self.snakes = None
        self.fruit = None
        self.initial_snake_length = config['initial_snake_length']
        self.rewards = config['rewards']
        self.max_step_limit = config.get('max_step_limit', 1000)
        self.is_game_over = False

        self.timestep_index = 0
        self.current_actions = None
        self.stats = [EpisodeStatistics() for i in range(2)]
        self.verbose = verbose
        self.debug_file = None
        self.stats_file = None

    def seed(self, value):
        """ Initialize the random state of the environment to make results reproducible. """
        random.seed(value)
        np.random.seed(value)

    @property
    def observation_shape(self):
        """ Get the shape of the state observed at each timestep. """
        return self.field.size, self.field.size

    @property
    def num_actions(self):
        """ Get the number of actions the agent can take. """
        return len(ALL_SNAKE_ACTIONS)

    def new_episode(self):
        """ Reset the environment and begin a new episode. """
        self.field.create_level()
        for i in range(2):
            self.stats[i].reset()
        self.timestep_index = 0

        self.snakes = [Snake(self.field.find_snake_head(20), length=self.initial_snake_length), Snake(self.field.find_snake_head(21), length=self.initial_snake_length)]
        self.field.place_snake(self.snakes)
        self.generate_fruit()
        self.current_actions = None
        self.is_game_over = False

        result = [TimestepResult(
            observation=self.get_observation(),
            reward=0,
            is_episode_end=self.is_game_over
        ) for i in range(2)]

        for i in range(2):
            self.record_timestep_stats(result[i])
        return result

    def record_timestep_stats(self, result):
        """ Record environment statistics according to the verbosity level. """
        timestamp = time.strftime('%Y%m%d-%H%M%S')

        # Write CSV header for the stats file.
        if self.verbose >= 1 and self.stats_file is None:
            self.stats_file = open(f'snake-env-{timestamp}.csv', 'w')
            # ryen
            stats_csv_header_line = self.stats[0].to_dataframe()[:0].to_csv(index=None)
            print(stats_csv_header_line, file=self.stats_file, end='', flush=True)

        # Create a blank debug log file.
        if self.verbose >= 2 and self.debug_file is None:
            self.debug_file = open(f'snake-env-{timestamp}.log', 'w')

        # ryen
        self.stats[0].record_timestep(self.current_actions, result)
        self.stats[0].timesteps_survived = self.timestep_index

        if self.verbose >= 2:
            print(result, file=self.debug_file)

        # Log episode stats if the appropriate verbosity level is set.
        if result.is_episode_end:
            if self.verbose >= 1:
                # ryen
                stats_csv_line = self.stats[0].to_dataframe().to_csv(header=False, index=None)
                print(stats_csv_line, file=self.stats_file, end='', flush=True)
            if self.verbose >= 2:
                print(self.stats, file=self.debug_file)

    def get_observation(self):
        """ Observe the state of the environment. """
        return np.copy(self.field._cells)

    def choose_action(self, actions):
        """ Choose the action that will be taken at the next timestep. """

        self.current_actions = actions
        for i in range(len(actions)):
            if actions[i] == SnakeAction.TURN_LEFT:
                self.snakes[i].turn_left()
            elif actions[i] == SnakeAction.TURN_RIGHT:
                self.snakes[i].turn_right()

    def timestep(self):
        """ Execute the timestep and return the new observable state. """

        self.timestep_index += 1
        rewards = [0,0]
        old_heads= []
        old_tails = []
        results = []
        for i in range(2):
            old_heads.append(self.snakes[i].head)
            old_tails.append(self.snakes[i].tail)

            # Are we about to eat the fruit?
            if self.snakes[i].peek_next_move() == self.fruit:
                self.snakes[i].grow()
                self.generate_fruit()
                old_tails[i] = None
                rewards[i] += self.rewards['ate_fruit'] * self.snakes[i].length
                self.stats[i].fruits_eaten += 1

            # If not, just move forward.
            else:
                if i == 0 or i == 1:
                    self.snakes[i].move()
                    rewards[i] += self.rewards['timestep']

            self.field.update_snake_footprint(old_heads[i], old_tails[i], self.snakes[i].head, i)

            # Hit a wall or own body?
            print(self.field)
            print(self.get_observation())
            if not self.is_alive(i):
                if self.has_hit_wall(i):
                    self.stats[i].termination_reason = 'hit_wall'
                    print("snake %d hit wall" % i)
                if self.has_hit_own_body(i):
                    print("snake %d hit own body" % i)
                    self.stats[i].termination_reason = 'hit_own_body'
                if self.has_hit_other_body(i):
                    print("snake %d hit other body" % i)
                    self.stats[i].termination_reason = 'hit_other_body'

                self.field[self.snakes[i].head] = 20 if i == 0 else 21 #CellType.SNAKE_HEAD
                self.is_game_over = True
                rewards[i] = self.rewards['died']

            # Exceeded the limit of moves?
            if self.timestep_index >= self.max_step_limit:
                self.is_game_over = True
                self.stats.termination_reason = 'timestep_limit_exceeded'

            results.append(TimestepResult(
                observation=self.get_observation(),
                reward=rewards[i],
                is_episode_end=self.is_game_over
            ))

            self.record_timestep_stats(results[i])
        return results
    

    def generate_fruit(self, position=None):
        """ Generate a new fruit at a random unoccupied cell. """
        if position is None:
            position = self.field.get_random_empty_cell()
        self.field[position] = CellType.FRUIT
        self.fruit = position

    def has_hit_wall(self, i):
        """ True if the snake has hit a wall, False otherwise. """
        return self.field[self.snakes[i].head] == CellType.WALL

    def has_hit_own_body(self, i):
        """ True if the snake has hit its own body, False otherwise. """
        if i == 0:
            return self.field[self.snakes[i].head] == CellType.SNAKE_BODY0 #or self.field[self.snakes[i].head] == CellType.SNAKE_HEAD0
        else:
            return self.field[self.snakes[i].head] == CellType.SNAKE_BODY1 #or self.field[self.snakes[i].head] == CellType.SNAKE_HEAD1

    def has_hit_other_body(self, i):
        if i == 0:
            return self.field[self.snakes[i].head] == CellType.SNAKE_BODY1 or self.field[self.snakes[i].head] == CellType.SNAKE_HEAD1 or len(set(list(collections.deque(self.snakes[i].body))).intersection(list(collections.deque(self.snakes[i+1].body)))) > 0
        else:
            return self.field[self.snakes[i].head] == CellType.SNAKE_BODY0 or self.field[self.snakes[i].head] == CellType.SNAKE_HEAD0 or len(set(list(collections.deque(self.snakes[i].body))).intersection(list(collections.deque(self.snakes[i-1].body)))) > 0

    def is_alive(self, i):
        """ True if the snake is still alive, False otherwise. """
        return not self.has_hit_wall(i) and not self.has_hit_own_body(i) and not self.has_hit_other_body(i)


class TimestepResult(object):
    """ Represents the information provided to the agent after each timestep. """

    def __init__(self, observation, reward, is_episode_end):
        self.observation = observation
        self.reward = reward
        self.is_episode_end = is_episode_end

    def __str__(self):
        field_map = '\n'.join([
            ''.join(str(cell) for cell in row)
            for row in self.observation
        ])
        return f'{field_map}\nR = {self.reward}   end={self.is_episode_end}\n'


class EpisodeStatistics(object):
    """ Represents the summary of the agent's performance during the episode. """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Forget all previous statistics and prepare for a new episode. """
        self.timesteps_survived = 0
        self.sum_episode_rewards = 0
        self.fruits_eaten = 0
        self.termination_reason = None
        self.action_counter = {
            action: 0
            for action in ALL_SNAKE_ACTIONS
        }

    def record_timestep(self, action, result):
        """ Update the stats based on the current timestep results. """
        self.sum_episode_rewards += result.reward
        if action is not None:
            #self.action_counter[action] += 1
            pass

    def flatten(self):
        """ Format all episode statistics as a flat object. """
        flat_stats = {
            'timesteps_survived': self.timesteps_survived,
            'sum_episode_rewards': self.sum_episode_rewards,
            'mean_reward': self.sum_episode_rewards / self.timesteps_survived if self.timesteps_survived else None,
            'fruits_eaten': self.fruits_eaten,
            'termination_reason': self.termination_reason,
        }
        flat_stats.update({
            f'action_counter_{action}': self.action_counter.get(action, 0)
            for action in ALL_SNAKE_ACTIONS
        })
        return flat_stats

    def to_dataframe(self):
        """ Convert the episode statistics to a Pandas data frame. """
        return pd.DataFrame([self.flatten()])

    def __str__(self):
        return pprint.pformat(self.flatten())
