import numpy as np
import torch

class Experience():
    def __init__(self, observation, action, reward, next_observation, done) -> None:
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation
        self.done = done

class ReplayBuffer():

    def __init__(self, min_size:int, max_size:int, batch_size:int, device) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self.batch_size = batch_size
        self.device = device
        self.experiences = []
        
    def add_experience(self, observation, action, reward, next_observation, done) -> None:
        experience = Experience(observation, action, reward, next_observation, done)
        if len(self.experiences) > self.max_size:
            del self.experiences[0]
        self.experiences.append(experience)

    def get_batch(self) -> list[Experience]:
        if len(self.experiences) < self.min_size:
            return None
        
        batch = np.random.choice(self.experiences, size = self.batch_size)

        non_final_mask = torch.tensor(tuple(map(lambda s: s.next_observation is not None, batch)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([torch.tensor(np.array([[s.next_observation]])) for s in batch if s.next_observation is not None]).to(self.device, dtype=torch.float)
        state_batch = torch.cat([torch.tensor(np.array([[s.observation]])) for s in batch]).to(self.device, dtype=torch.float)
        action_batch = torch.cat([torch.tensor([[s.action.id]]) for s in batch]).to(self.device)
        reward_batch = torch.tensor([s.reward for s in batch]).to(self.device)

        return non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch