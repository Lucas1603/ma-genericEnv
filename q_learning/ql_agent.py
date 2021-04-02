import numpy as np


class Agent:
    """Class that refers to the q-learning agent 
    
    Args:
        state ( list ): The initial state
        state_space ( Box ): The state space of the environment
        action_space ( Discrete ): The action space of the environment
        alpha ( float ): Alpha learning rate
        gamma ( float ): Gamma discount factor
        exploration_strategy ( fn ): The exploration strategy

    Methods:
        chooseAction:
            Chooses an action according to the exploration strategy.
            Args:
                None
            Returns:
                int: The chosen action.
        learn:
            Does the update on the q-value.
            Args:
                next_state ( list ): The state that was transitioned to.
                reward ( int/float ): The reward got.
                done (bool)[default: False]: Wheter the agent is done.
            Returns:

    """
    def __init__( self, 
                  state, 
                  state_space, 
                  action_space, 
                  alpha, 
                  gamma, 
                  exploration_strategy ):

        # initialize the state
        self.state = state
        # stores the properties of the state and action space
        self.state_space = state_space
        self.action_space = action_space

        # initialize the action variable
        self.action = None

        # and the alpha and gamma parameters
        self.alpha = alpha
        self.gamma = gamma

        # the q-table, starting only with zeros for the current state
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        
        # sets the exploration strategy
        self.exploration = exploration_strategy
        
        # initialize the variable that will keep the total amount of reward
        self.total_reward = 0
    
    def choose_action(self):
        # choose the action according to the epsilon-greedy policy
        self.action = self.exploration.choose( self.q_table, self.state, self.action_space )
        return self.action
    
    def learn(self, next_state, reward, done=False):
        # if the state was never reached before
        if next_state not in self.q_table:
            # add this state to the q-table
            self.q_table[next_state] = [ 0 for _ in range( self.action_space.n ) ]
        

        s = self.state
        s_prime = next_state
        action = self.action

        # update the q-value
        td_target = reward + self.gamma * max( self.q_table[s_prime] )
        td_delta = td_target - self.q_table[s][action]
        self.q_table[s][action] += self.alpha * td_delta

        # update the state
        self.state = s_prime
        # sum the current reward to the total amount 
        self.total_reward += reward 