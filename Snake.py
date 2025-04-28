import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque


class SnakeEnv(gym.Env):
    """A simple 10×10 Snake environment compatible with Gymnasium RL algorithms.

    Observation space
    -----------------
    Box(low=0, high=3, shape=(10, 10), dtype=np.int8)
        0 – empty cell
        1 – snake body
        2 – snake head
        3 – food

    Action space
    ------------
    Discrete(4)
        0 – up    (−1, 0)
        1 – down  (+1, 0)
        2 – left  (0, −1)
        3 – right (0, +1)

    Reward
    ------
    +10   for eating food
    −0.1  living penalty each step (encourages efficiency)
    −11   for crashing into wall or self (terminates episode)
    +30   bonus for completely filling the grid (win condition)

    Termination
    -----------
    * Collision with wall or body (terminated=True)
    * Snake length == grid_size² (terminated=True)
    * Reaching max_steps (truncated=True) – avoids infinite games
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size: int = 10, render_mode: str | None = None):
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(grid_size, grid_size), dtype=np.int8
        )

        self.render_mode = render_mode
        self.np_random = np.random.RandomState()

        # Direction deltas: up, down, left, right
        self._directions = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }
        # Conservative upper‑bound on episode length
        self._max_steps = grid_size * grid_size * 4

        # Internal state placeholders
        self._grid: np.ndarray | None = None
        self._snake: deque[tuple[int, int]] | None = None
        self._direction: int | None = None
        self._food_pos: tuple[int, int] | None = None
        self._steps: int = 0

    # ---------------------------------------------------------------------
    # Environment helper methods
    # ---------------------------------------------------------------------
    def _place_food(self) -> None:
        """Randomly place food on an empty cell; assumes at least 1 empty cell."""
        empty = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self._grid[r, c] == 0
        ]
        if not empty:
            return  # grid is full – handled elsewhere
        r, c = random.choice(empty)
        self._grid[r, c] = 3
        self._food_pos = (r, c)

    # ---------------------------------------------------------------------
    # Core Gymnasium API
    # ---------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random.seed(seed)
            random.seed(seed)

        self._grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self._direction = 3  # start moving right
        mid = self.grid_size // 2
        # Tail first, head last
        self._snake = deque([(mid, mid - 1), (mid, mid)])
        for r, c in list(self._snake)[:-1]:
            self._grid[r, c] = 1  # body
        head_r, head_c = self._snake[-1]
        self._grid[head_r, head_c] = 2  # head

        self._place_food()
        self._steps = 0
        return self._grid.copy(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action provided"

        # Disallow 180‑degree reversals
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}
        if action != opposite[self._direction]:
            self._direction = action

        dr, dc = self._directions[self._direction]
        head_r, head_c = self._snake[-1]
        new_r, new_c = head_r + dr, head_c + dc

        self._steps += 1
        reward = -0.1  # living penalty
        terminated = False
        truncated = False

        # Collision check ---------------------------------------------------
        if not (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size):
            # Hit wall
            reward = -11
            terminated = True
        elif self._grid[new_r, new_c] == 1:
            # Hit own body
            reward = -11
            terminated = True
        else:
            ate_food = self._grid[new_r, new_c] == 3

            # Move the snake's head
            self._snake.append((new_r, new_c))
            self._grid[new_r, new_c] = 2
            # Old head becomes body
            self._grid[head_r, head_c] = 1

            if ate_food:
                reward = 10
                # Win condition – filled board
                if len(self._snake) == self.grid_size * self.grid_size:
                    reward += 30
                    terminated = True
                else:
                    self._place_food()
            else:
                # Standard move – pop tail
                tail_r, tail_c = self._snake.popleft()
                self._grid[tail_r, tail_c] = 0

        # Truncation to avoid endless games
        if self._steps >= self._max_steps:
            truncated = True

        return self._grid.copy(), reward, terminated, truncated, {}

    # ---------------------------------------------------------------------
    # Rendering utilities
    # ---------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        symbols = {0: " . ", 1: " o ", 2: " O ", 3: " X "}
        print("\n".join("".join(symbols[cell] for cell in row) for row in self._grid))
        print()

    def close(self):
        pass


# -------------------------------------------------------------------------
# Registration helper (optional)
# -------------------------------------------------------------------------

def register_snake_env(id: str = "Snake‑10x10‑v0") -> None:
    """Register the environment with Gymnasium so you can call gym.make(id)."""
    from gymnasium.envs.registration import register

    if id in gym.registry.keys():
        # Avoid duplicate registration during notebook reloads
        return

    register(
        id=id,
        entry_point="snake_env:SnakeEnv",
        max_episode_steps=400,
    )

