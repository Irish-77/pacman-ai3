import numpy as np

class Experience():
    def __init__(self, observation, action, reward, next_observation, done) -> None:
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation
        self.done = done

    def __iter__(self):
        yield from [self.observation, self.action, self.reward, self.next_observation, self.done]

class ReplayBuffer():

    def __init__(self, min_size:int, max_size:int, batch_size:int, gamma:float=0.99) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.experiences = []

    def add_experience(self, observation, action, reward, next_observation, done) -> None:
        # check len
        experience = Experience(observation, action, reward, next_observation, done)
        if len(self.experiences) > self.max_size:
            del self.experiences[0]
        self.experiences.append(experience)

    def get_batch(self):
        # customize if necessary

        if len(self.experiences) < self.min_size:
            return None
        
        selected_experiences = np.random.choice(self.experiences, size = self.batch_size)

        observations, actions, rewards, next_observations, dones = [], [], [], [], []

        for observation, action, reward, next_observation, done in selected_experiences:
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_observation)
            dones.append(done)

        return observations, actions, rewards, next_observations, dones