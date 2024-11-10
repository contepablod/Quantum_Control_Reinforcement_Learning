import random
from collections import deque

# Replay Memory
class ReplayMemory:
    """
    A class to represent replay memory for storing experiences
    during reinforcement learning.

    Attributes:
    ----------
    memory : deque
        A deque to store the experiences with a fixed maximum length.
    """

    def __init__(self, capacity):
        """
        Initializes the replay memory with a specified capacity.

        Parameters:
        ----------
        capacity : int
            The maximum number of experiences the memory can hold.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Adds an experience to the replay memory.

        Parameters:
        ----------
        state : np.ndarray
            The state observed before taking the action.
        action : int
            The action taken by the agent.
        reward : float
            The reward received after taking the action.
        next_state : np.ndarray
            The state observed after taking the action.
        done : bool
            Whether the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the replay memory.

        Parameters:
        ----------
        batch_size : int
            The number of experiences to sample.

        Returns:
        -------
        list
            A list of sampled experiences.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Returns the current size of the replay memory.

        Returns:
        -------
        int
            The number of experiences currently stored in the replay memory.
        """
        return len(self.memory)
