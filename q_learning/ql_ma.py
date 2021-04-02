from arg_parser import ArgumentParser
from exploration.epsilon_greedy import EpsilonGreedy
from ql_agent import Agent
import gym


def discretize_switch(state):
    new_state = []
    for obs in state:
        new_state.append( int(obs[0] * 2) * 7 + round( obs[1]*6 ) )

    return new_state


if __name__ == '__main__':

    # get the arguments from the command line
    alpha, gamma, epsilon, min_epsilon, decay, runs = ArgumentParser().get_arguments()

    # import the environment or raise an error
    try:
        env = gym.make('ma_gym:Switch2-v0')
    except:
        raise ImportError


    # get the initial state
    state = env.reset()
    # discretize the state
    state = discretize_switch(state)

    # instatiate the agents
    ql_agents = [
        Agent( state[i], 
               env.observation_space[i], 
               env.action_space[i], 
               alpha, 
               gamma, 
               EpsilonGreedy(epsilon, min_epsilon, decay) 
        ) for i in range(2)
    ]

    while True:

        # get the initial state
        state = env.reset()

        # discretize the state
        state = discretize_switch(state)
    
        # initialize the dones for every agent
        done = [False for _ in ql_agents]

        while not all(done):
            actions = [ agent.choose_action() for agent in ql_agents ]
            print(actions)

            s_prime, reward, done, _ = env.step(actions)
            s_prime = discretize_switch(s_prime)

            for i in range(len(ql_agents)):
                ql_agents[i].learn( s_prime[i], reward[i] )
            
            env.render()
