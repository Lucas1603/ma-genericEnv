import numpy as np
from gym import spaces

class EpsilonGreedy:
    """Epsilon Greedy Policy class.

    Args:
        initial_epsilon ( float ): The epsilon that the algorithm will start with.
        min_epsilon ( float ): The minimum value that epsilon can reach.
        decay ( float ): The decay factor for epsilon.
    
    Methods:
        choose:
            Function responsible for choosing an action for the agent.
            Args:
                q_table ( numpy.array ): The table that stores the q-value for every state-action pair.
                state ( numpy.array ): The state that the agent is in.
                action_space ( Discrete ): The possible actions.
            Returns:
                ( int ): the action that the agent will take.
        reset:
            Args:

            Returns:
    """

    def __init__(self, initial_epsilon, min_epsilon, decay):
        # set the variables
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def choose(self, q_table, state, action_space):
        """Function responsible for choosing an action for the agent.

        Args:
            q_table ( numpy.array ): The table that stores the q-value for every state-action pair.
            state ( numpy.array ): The state that the agent is in.
            action_space ( Discrete ): The possible actions.
        
        Returns:
            ( int ): the action that the agent will take.
        """
        
        if np.random.rand() < self.epsilon:
            action = int(action_space.sample())
        else:
            action = np.argmax(q_table[state])
        # print(action)

        self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)
        #print(self.epsilon)

        return action

    def reset(self):
        self.epsilon = self.initial_epsilon