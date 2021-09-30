import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

running_average = lambda m_n_1, r_i, n: m_n_1 + ((r_i - m_n_1) / n)


class RewardDistribution:
    def __init__(self, k=10):
        self.k = k
        self.mu = 0
        self.sigma = 1
        self.q_star_mu = np.random.normal(self.mu, self.sigma, k)
        self.q_star_sd = np.ones(k)

    def get_reward(self, action):
        Rt = np.random.normal(self.q_star_mu[action], self.q_star_sd[action], 1)
        return Rt

    def plot(self):
        # create a data frame to plot the distribution
        df = {}
        sample_size = 1000
        for action in range(self.k):
            mu = self.q_star_mu[action]
            sd = self.q_star_sd[action]
            df[f'action_{action}'] = np.random.normal(mu, sd, sample_size)
        df = pd.DataFrame(data=df)
        sns.boxplot(data=df)


class EpsBandit:
    def __init__(self):
        self.eps = None
        self.k = None
        self.qa = None
        self.actions = None
        self.rd = None

    def set_epsilon(self, eps):
        pass

    def set_reward_distribution(self, rd):
        pass

    def set_k_arms(self, k):
        pass

    def get_action_dist(self):
        pass

    def get_q_values(self):
        pass

    def _sample_an_action(self):

        def greedy_action():
            # pick action corresponding to high qa
            return np.argmax(self.qa)

        def random_action():
            # pick random action from k selections
            return np.random.choice(self.k)

        if self.eps == 0:
            # always greedy choice
            return greedy_action()
        else:
            p = np.random.rand()
            # high epsilon means more weight to random actions
            if p < self.eps:
                return random_action()
            else:
                return greedy_action()

    def _execute_an_action(self, action):
        sampled_rewards = self.rd.get_reward(action=action)
        self.actions[action] += 1
        return sampled_rewards


class Experiment(EpsBandit):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.total_avg_reward = 0.0
        self.average_rewards = {}

    def set_epsilon(self, eps):
        self.eps = eps

    def set_k_arms(self, k):
        self.k = k
        self.qa = np.zeros(k)
        self.actions = np.zeros(k)

    def set_reward_distribution(self, rd):
        self.rd = rd

    def get_name(self):
        return self.name

    def get_average_rewards(self):
        return pd.DataFrame(self.average_rewards)

    def get_action_dist(self):
        dataframe = {}
        for _index, value in enumerate(self.actions):
            dataframe[f'a{_index}'] = [value]
        return pd.DataFrame(dataframe)

    def get_q_values_dist(self):
        dataframe = {}
        for _index, value in enumerate(self.qa):
            dataframe[f'q_a{_index}'] = [value]
        return pd.DataFrame(dataframe)

    def run(self, steps=10):
        for t in range(1, steps):
            action = self._sample_an_action()
            r_t = self._execute_an_action(action)
            self.total_avg_reward = running_average(m_n_1=self.total_avg_reward, r_i=r_t, n=t)
            self.qa[action] = running_average(m_n_1=self.qa[action], r_i=r_t, n=self.actions[action])
            self.average_rewards[f't{t}'] = [round(float(self.total_avg_reward), 2)]


class Experiments:
    def __init__(self):
        self.time_steps = 10
        self.experiments = []

    def set_time_steps(self, steps=10):
        self.time_steps = steps

    def add_an_experiment(self, experiment):
        self.experiments.append(experiment)

    def run_all_experiments(self):
        for _experiment in self.experiments:
            _experiment.run(steps=self.time_steps)
            print(f'Finished Running experiment {_experiment}')

    def get_results(self):
        results = dict()
        results['time_step'] = []
        results['epsilon'] = []
        results['average_reward'] = []
        for t in range(0, self.time_steps):
            for _exp in self.experiments:
                _res = _exp.get_average_rewards()
                results['time_step'].append(f't{t}')
                results['epsilon'].append(_exp.get_name())
                if t == 0:
                    results['average_reward'].append(0.0)
                else:
                    results['average_reward'].append(_res.loc[[0], [f't{t}']].values[0][0])
        return pd.DataFrame(results)


number_of_arms = 10
# One reward distribution comman across all
reward_distribution = RewardDistribution(k=number_of_arms)
experiments = Experiments()
experiments.set_time_steps(steps=1000)

experiment_1 = Experiment(name='eps_0')
experiment_1.set_k_arms(k=number_of_arms)
experiment_1.set_reward_distribution(rd=reward_distribution)
experiment_1.set_epsilon(eps=0.0)
experiments.add_an_experiment(experiment=experiment_1)

experiment_2 = Experiment(name='eps_0_0_1')
experiment_2.set_k_arms(k=number_of_arms)
experiment_2.set_reward_distribution(rd=reward_distribution)
experiment_2.set_epsilon(eps=0.01)
experiments.add_an_experiment(experiment=experiment_2)

experiment_3 = Experiment(name='eps_0_1')
experiment_3.set_k_arms(k=number_of_arms)
experiment_3.set_reward_distribution(rd=reward_distribution)
experiment_3.set_epsilon(eps=0.1)
experiments.add_an_experiment(experiment=experiment_3)


net_result = None
flag = False
for episode in range(0, 100):
    print(f"Running episode {episode}")
    experiments.run_all_experiments()
    results = experiments.get_results()
    if not flag:
        net_result = results
        flag = True
    else:
        net_result['average_reward'] = running_average(m_n_1=net_result['average_reward'],
                                                       r_i=results['average_reward'],
                                                       n=episode)
print(net_result.head())
sns.lineplot(data=net_result, x='time_step', y='average_reward', hue='epsilon')
plt.show()

#rint(experiment_1.get_action_dist())
#print(experiment_1.get_q_values_dist())

