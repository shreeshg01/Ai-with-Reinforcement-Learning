import numpy as np
import matplotlib.pyplot as plt

class ReinforcementLearning:
    def __init__(self, mdp, sampleReward):
        """
        Constructor for the RL class

        :param mdp: Markov decision process (T, R, discount)
        :param sampleReward: Function to sample rewards (e.g., Bernoulli, Gaussian). This function takes one argument:
        the mean of the distribution and returns a sample from the distribution.
        """
        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self, state, action):
        '''
        Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs:
        reward -- sampled reward
        nextState -- sampled next state
        '''
        reward = self.sampleReward(self.mdp.R[action, state])
        cumProb = np.cumsum(self.mdp.T[action, state, :])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward, nextState]

    def reinforce(self, theta=None, alpha=0.01, gamma=0.99, n_episodes=3000):
        """
        REINFORCE Algorithm Implementation
        :param theta: Initial policy parameters (shape = [|A|, |S|])
        :param alpha: Learning rate
        :param gamma: Discount factor
        :param n_episodes: Number of episodes
        :return: Cumulative rewards and final policy parameters
        """
        if theta is None:
            theta = np.random.rand(self.mdp.R.shape[0], self.mdp.R.shape[1])

        def softmax_policy(state):
            preferences = theta[:, state]
            exp_preferences = np.exp(preferences - np.max(preferences))
            return exp_preferences / np.sum(exp_preferences)

        cum_rewards = []

        for episode in range(n_episodes):
            state = 0  # Start from the initial state
            trajectory = []  # To store (state, action, reward) tuples
            G = 0  # Cumulative reward

            while True:
                policy = softmax_policy(state)
                action = np.random.choice(range(len(policy)), p=policy)
                reward, next_state = self.sampleRewardAndNextState(state, action)
                trajectory.append((state, action, reward))
                G += reward
                state = next_state
                if state == self.mdp.R.shape[1] - 1:  # Terminal state
                    break

            cum_rewards.append(G)

            # Update policy parameters
            for t, (state, action, reward) in enumerate(trajectory):
                G_t = sum([r * (gamma ** i) for i, (_, _, r) in enumerate(trajectory[t:])])
                policy = softmax_policy(state)
                grad_ln_pi = np.zeros_like(theta)
                grad_ln_pi[action, state] = 1 - policy[action]
                theta += alpha * grad_ln_pi * G_t

        return cum_rewards, theta

    def actorCritic(self, theta=None, alpha_theta=0.01, alpha_v=0.01, gamma=0.99, n_episodes=3000):
        """
        Actor-Critic Algorithm Implementation
        :param theta: Initial policy parameters (shape = [|A|, |S|])
        :param alpha_theta: Learning rate for policy parameters
        :param alpha_v: Learning rate for value function
        :param gamma: Discount factor
        :param n_episodes: Number of episodes
        :return: Cumulative rewards, final policy parameters, and value function
        """
        if theta is None:
            theta = np.random.rand(self.mdp.R.shape[0], self.mdp.R.shape[1])
        V = np.zeros(self.mdp.R.shape[1])  # Value function initialization

        def softmax_policy(state):
            preferences = theta[:, state]
            exp_preferences = np.exp(preferences - np.max(preferences))
            return exp_preferences / np.sum(exp_preferences)

        cum_rewards = []

        for episode in range(n_episodes):
            state = 0  # Start from the initial state
            G = 0  # Cumulative reward

            while True:
                policy = softmax_policy(state)
                action = np.random.choice(range(len(policy)), p=policy)
                reward, next_state = self.sampleRewardAndNextState(state, action)
                G += reward

                # Compute TD error
                td_error = reward + gamma * V[next_state] - V[state]

                # Update value function
                V[state] += alpha_v * td_error

                # Update policy parameters
                grad_ln_pi = np.zeros_like(theta)
                grad_ln_pi[action, state] = 1 - policy[action]
                theta += alpha_theta * grad_ln_pi * td_error

                state = next_state
                if state == self.mdp.R.shape[1] - 1:  # Terminal state
                    break

            cum_rewards.append(G)

        return cum_rewards, theta, V

# Testing the implementation
if __name__ == "__main__":
    from MDP import build_mazeMDP

    mdp = build_mazeMDP(b=0.1)
    rl = ReinforcementLearning(mdp, np.random.normal)

    n_episode = 100
    n_trials = 2

    # Test REINFORCE
    out = np.zeros([n_trials, n_episode])
    for i in range(n_trials):
        cum_rewards, theta = rl.reinforce()
        out[i, :] = np.array(cum_rewards)
    plt.plot(out.mean(axis=0), label='REINFORCE')

    # Test Actor-Critic
    out = np.zeros([n_trials, n_episode])
    for i in range(n_trials):
        cum_rewards, policy_ac, v = rl.actorCritic()
        out[i, :] = cum_rewards
    plt.plot(out.mean(axis=0), label='Actor-Critic')

    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Rewards")
    plt.legend()
    plt.title("Performance Comparison of REINFORCE and Actor-Critic")
    plt.show()
