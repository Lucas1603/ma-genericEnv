from gym import Env
from gym.spaces import Box, Discrete
import numpy as np
import random

class GenericEnv(Env):
    """Environment to act as a blueprint for others.

    Args:
        n_agents ( int ): The number of agents that will act in the environment.
        action_space_n ( int ): The number of actions possible.
        obs_space_low ( list ): The lowest values possible for a state.
        obs_space_high ( list ): The highest values possible for a state.

    Methods:
        step:
            Function responsible for taking an action.
            Args:
                actions ( list(int) ): A list of actions, one per agent.
            Returns:
                ( numpy.array ): The state after the action.
                ( list(int) ): A list of all rewards, one per agent.
                ( list(bool) ): A list that indicates wheter the agent is done or still can take actions.
                ( dict ): A dictionary containing some informations about the environment.
        reset:
            Put random initial values in the state
            Args: 
                None
            Returns: 
                None
        render:
            Args: 
                None
            Returns: 
                None
    """
    def __init__(   self, 
                    n_agents,
                    action_space_n,
                    obs_space_low, obs_space_high,
                    ):

        # where we define how the actions and observation states will be
            # Discrete      -> Discrete(3)
            # Box           -> Box( low=[], high=[] )

        self._check_val(n_agents, action_space_n, obs_space_low, obs_space_high )

        self.n_agents = n_agents
        self.action_space_n = action_space_n
        self.obs_space_low = obs_space_low 
        self.obs_space_high = obs_space_high

        # variable that will save if the agents are done, and thus, must not interact with the environment
        self.dones = [False] * n_agents

        # create the action space according to the number of agents
        self.action_space = np.array( [ Discrete( self.action_space_n ) ] * self.n_agents ) 
        
        # create the observation space
        self.observation_space =   np.array( 
            [ 
                Box( low= np.float32( self.obs_space_low ), high= np.float32( self.obs_space_high ) ) 
            ]  * self.n_agents 
        )

        # initialize the state
        self.reset()
        
    
    def step(self, actions):
        """Function responsible for taking an action.

        Args:
            actions ( list(int) ): A list of actions, one per agent.

        Returns:
            ( numpy.array ): The state after the action.
            ( list(int) ): A list of all rewards, one per agent.
            ( list(bool) ): A list that indicates wheter the agent is done or still can take actions.
            ( dict ): A dictionary containing some informations about the environment.
        """
        # make sure that there is one action per agent
        assert len(actions) == self.n_agents, "There are more - or less - actions than expected"

        # determine the array that will keep the next state
        next_states = []
        
        # calculate a random state to be the next - temporarily
        np_obs_low = np.array(self.obs_space_low)
        np_random = np.array( [random.random() for _ in range( len(self.obs_space_low) )] )
        np_obs_high = np.array( self.obs_space_high )
        
        aux_state = [ np_obs_low +  np_random * np_obs_high for _ in self.observation_space ]

        # update the state only if the agent is not done
        for i, action in enumerate(actions):
            if self.dones[i] == True:
                next_states.append( self.states[i] )
            else:
                next_states.append( aux_state[i] )
        
        # calculate the rewards
        rewards = self._reward_function( self.states, next_states, actions  )
        # calculate if the agents are done
        dones = self._check_done( next_states )

        # additional infos
        infos = {}

        # update the main states
        self.states = next_states
        
        return self.states, rewards, dones, infos


    def render(self):
        pass

    
    def reset(self):
        """Put random initial values in the state

        Args:
            None

        Returns:
            None
        """
        np_obs_low = np.array(self.obs_space_low)
        np_random = np.array( [random.random() for _ in range(len(self.obs_space_low))] )
        np_obs_high = np.array( self.obs_space_high )
        
        self.states = [ np_obs_low +  np_random * np_obs_high for _ in self.observation_space ]

   
    def _check_done(self, next_states):
        """Function for checking if the agent is done

        Args:
            next_states ( numpy.array(int) ): The state after the actions have been taken. 

        Returns:
            ( list(bool) ): A list that indicates wheter the agent is done or still can take actions.
        """

        # determine randomly if the agent is done - temporarily
        
        for i in range( self.n_agents ):
            # if the agent is already done, do nothing
            if self.dones[i]:
                continue
            # otherwise, check if it's done
            self.dones[i] = np.random.choice( [True, False], 1, p=[0.1,0.9] ) 

        return self.dones

    
    def _reward_function(self, current_states, next_states, actions ):
        """Function where all the rewards are going to be assigned, depending on the action and state
        """
        # random rewards - temporarily
        return [ random.randint(-1,1) for _ in range(self.n_agents) ]
        
    
    def _check_val(self, n_agents, action_space_n, obs_space_low, obs_space_high):
        """ Check if the inputs are allowed
        """
        # you have to have at least one agent
        assert (n_agents > 0), "The number of agents must be greater than zero"
        # and at least one possible action
        assert (action_space_n > 0), "The action space must be greater than zero"
        # the shapes of obs_space_low and obs_space_high must be equal
        assert ( np.array( obs_space_low ).shape == np.array( obs_space_high ).shape ), "Make sure that the shape of the maximum observation and minimum are equal"



if __name__ == '__main__':
    # some tests
    env = GenericEnv( 4, 4, [0,0, 0], [100,100,100] )
    while not all( env.dones ):
        actions = [ action.sample() for action in env.action_space ]
        env.step( actions )
