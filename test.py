from pickle import TRUE
from environment import Environment
from recorder import Recorder
from replay_buffer import ReplayBuffer
from model import DQModelWithCNN
from agent import Agent
from itertools import count
from os.path import join
import os
import glob
from pyparsing import empty
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def train() -> None:
    NUM_EPISODES = 1000
    TARGET_UPDATE = 25
    SAVE_STATE = 50
    BATCH_SIZE = 200
    MIN_REPLAY_SIZE = BATCH_SIZE
    MAX_REPLAY_SIZE = 1e4
    GAMMA = 0.99
    SHOW = False
    RECORD = True
    DEBUG = True
    id = 'test'

    writer = SummaryWriter(f'tensorboard/{id}')
    logger = init_logging(id, DEBUG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_agent = Agent(model=DQModelWithCNN(device).to(device))
    target_agent = Agent(model=DQModelWithCNN(device).to(device))
    replay_buffer = ReplayBuffer(min_size=MIN_REPLAY_SIZE,
        max_size=MAX_REPLAY_SIZE, batch_size=BATCH_SIZE, device=device)
    env = Environment(path=join('maps', 'map.csv'))
    optimizer = optim.RMSprop(train_agent.model.parameters())
    if RECORD:
        recorder = Recorder()

    for e in range(NUM_EPISODES):
        observation = env.reset()
        new_id = f'{id}_{e}'
        acc_reward = 0
        if RECORD and (e%SAVE_STATE == 0):
            recorder.init_new_video(id=new_id) 
        for i in count():
            action = train_agent.predict_new_move_eps(torch.tensor(np.array([[observation]]), dtype=torch.float))
            reward, next_observation, done = env.step(action)
            acc_reward += reward
            if SHOW:
                env.show()
            if RECORD and (e%SAVE_STATE == 0):
                env.write_img_to_buffer()
                recorder.add_image(env.output_buffer)
            if done:
                next_observation = None

            replay_buffer.add_experience(observation, action, reward,
                next_observation, done)

            observation = next_observation

            # training
            batch = replay_buffer.get_batch()
            losses = []
            if batch is not None:
                non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch = batch

                state_action_values = train_agent.model(state_batch).gather(1, action_batch)
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                next_state_values[non_final_mask] = target_agent.model(non_final_next_states).max(1)[0].detach()
                expected_state_action_values = (next_state_values * GAMMA) + reward_batch
                criterion = nn.SmoothL1Loss()

                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                for param in train_agent.model.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
                losses.append(loss)
            if done:
                break
        mean_loss = torch.mean(torch.stack(losses)) if len(losses) > 0 else 0
        writer.add_scalars('pacman', tag_scalar_dict={'episode': e, 'steps': i, 'avg_loss': mean_loss, 'acc_reward': acc_reward}, global_step=e)
        if (e % SAVE_STATE == 0):
            logger.info(f'episode: {e:6d} | #steps: {i:3d} | avg_loss: {mean_loss:8.5f} | acc_reward: {acc_reward:3d}')
            train_agent.model.save(id=new_id)
            if RECORD:
                recorder.close_recording()
        if e % TARGET_UPDATE == 0:
            target_agent.model.load_state_dict(train_agent.model.state_dict())

def init_logging(id:str, debug:bool = False) -> logging.Logger:
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger('pacman_rl')
    logger.setLevel(level=level)
    fh = logging.FileHandler(os.path.join('logs', f'pacman_{id}.log'), mode='a')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def get_latest_model_name() -> str:
    path = os.path.join(os.getcwd(), 'models', '*')
    list_of_files = glob.glob(path) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return os.path.basename(latest_file).split('.')[0]

def play(model_name:str = None) -> None:
    SHOW = True
    RECORD = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DQModelWithCNN()
    if model_name is None:
        model_name = get_latest_model_name()
        print(f'No model specified. Using latest model: {model_name}.')
    model.load(model_name)
    agent = Agent(model=model.to(device))

    env = Environment(path=join('maps', 'map.csv'))
    observation = env.reset()
    if RECORD:
        recorder = Recorder()
        recorder.init_new_video(f'play_{model_name}_{datetime.today().strftime("%Y_%m_%d_%H_%M_%S")}')

    for i in count():
        action = agent.predict_new_move(torch.tensor(np.array([[observation]]), dtype=torch.float))
        _, next_observation, done = env.step(action)
        observation = next_observation

        if SHOW:
            env.show()
        if RECORD:
            env.write_img_to_buffer()
            recorder.add_image(env.output_buffer)
        if done:
            break

    if RECORD:
        recorder.close_recording()



if __name__ == '__main__':
    # train()
    play()

# avg steps, avgavgloss, avg acc_reward
# aufteilen
    # strong type
    # docstrings --> Google format?
    # 