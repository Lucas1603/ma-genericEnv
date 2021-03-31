import argparse

class ArgumentParser:
    """Class used to parse the arguments given by command line.

    Methods:
        getArguments: 
            Returns all the parsed arguments 
            Args:
                None
            Returns:
                float: Alpha learning rate.
                float: Gamma discount rate.
                float: Epsilon.
                float: Minimum epsilon.
                float: Epsilon decay.
                int: Number of runs.
    """

    def __init__(self):
        self.prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Q-Learning""")

        self.prs.add_argument("--alpha", "-a", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
        self.prs.add_argument("--gamma", "-g", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
        self.prs.add_argument("--epsilon", "-e", type=float, default=0.05, required=False, help="Epsilon.\n")
        self.prs.add_argument("--min_epsilon", "-me", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
        self.prs.add_argument("--decay", "-d", type=float, default=1.0, required=False, help="Epsilon decay.\n")
        self.prs.add_argument("--runs", "-r", type=int, default=1, help="Number of runs.\n")
        
    def get_arguments(self):
        """Function responsible for returning all the parsed arguments

        Args:
            None

        Returns:
            float: Alpha learning rate.
            float: Gamma discount rate.
            float: Epsilon.
            float: Minimum epsilon.
            float: Epsilon decay.
            int: Number of runs.
        """
        args = self.prs.parse_args()
        
        alpha = args.alpha
        gamma = args.gamma
        epsilon = args.epsilon
        min_epsilon = args.min_epsilon
        decay = args.decay
        runs = args.runs

        return alpha, gamma, epsilon, min_epsilon, decay, runs