from gym import Env
from gym.spaces import Box, Discrete
import numpy as np
import random

class GenericEnv(Env):
    def __init__(   self, 
                    n_agents,
                    action_space_n,
                    obs_space_low, obs_space_high,
                    # step
                    # reward
                    ):

        # where we define how the action and observation state will be
            # Discrete      -> Discrete(3)
            # Box           -> Box( low=[], high=[] )

        self.check_val(n_agents, action_space_n, obs_space_low, obs_space_high )

        self.n_agents = n_agents
        self.action_space_n = action_space_n
        self.obs_space_low = obs_space_low 
        self.obs_space_high = obs_space_high

        # create the action space according to the number of agents
        self.action_space = np.array( [ Discrete( self.action_space_n ) ] * self.n_agents ) 
        
        # create the observation space
        self.observation_space =   np.array( [ Box(  low= np.float32( self.obs_space_low ), 
                                            high= np.float32( self.obs_space_high ) )
                                    ] * self.n_agents )

        # initialize the state
        self.reset()
        
    
    def step(self, actions):
        # make sure that there are one action per agent
        assert len(actions) == self.n_agents, "There are more - or less - actions than expected"
        
        next_states = [ np.array( [random.random() for _ in range(len(self.obs_space_low))] ) for _ in actions ]

        next_state = []
        for action in actions:
            next_states.append( np.array( [ random.random() for _ in self.obs_space_low ] ) )
            rewards = self.reward_function( self.states, next_states, actions  )
            # done = checkDone( next_state )

        self.states = next_states

        print(self.states)

        # return self.states, rewards, dones, infos

    def render(self):
        pass

    
    def reset(self):
         # put random initial values in the state
        np_obs_low = np.array(self.obs_space_low)
        np_random = np.array( [random.random() for _ in range(len(self.obs_space_low))] )
        np_obs_high = np.array( self.obs_space_high )
        
        self.states = [ np_obs_low +  np_random * np_obs_high for _ in self.observation_space ]
        print(self.states)


    
    def reward(self, current_states, next_states, actions ):
        pass
    
    
    def check_val(self, n_agents, action_space_n, obs_space_low, obs_space_high):
        # check if the inputs are allowed
        assert (n_agents > 0), "The number of agents must be greater than zero"
        assert (action_space_n > 0), "The action space must be greater than zero"


if __name__ == '__main__':

    env = GenericEnv( 2, 4, [0,0,0], [100,100,100] )
    actions = [ action.sample() for action in env.action_space] 
    env.step( actions )
    # print( actions )
