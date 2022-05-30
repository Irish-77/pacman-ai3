from environment import Environment
from recorder import Recorder
from itertools import count
from os.path import join
from pyparsing import empty


if __name__ == '__main__':
    env = Environment(path = join('maps', 'map.csv'))
    rec = Recorder()
    total_reward = 0
    env.show()
    for i in count():
        reward, _, done = env.step()
        env.show()
        total_reward += reward

        rec.add_image(env.output_buffer)
        if done:
            print(f'GAME OVER - total steps: {i} - reward {total_reward}')
            rec.close_recording()
            break


# Next steps:
# - Modell laden & speichern
# - Modell kopieren
# - Train methode
# -  