import io
import numpy as np
import matplotlib.colors as mat_colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from os.path import join
from ghost import Ghost
from coin import Coin
from pacman import Pacman
from options import Blocks, Movements

class Environment():

    def __init__(self, path = 'map.csv', save_img_as_buffer = True):
        self.map = self.load_map(path)
        self.ghosts = []
        self.coins = []
        self.pacman = None
        colors = [x.color for x in Blocks]
        self.cmap = mat_colors.ListedColormap(colors)


        self.max_coins = 0
        self.max_steps = 1000
        self.current_steps = 0

        self.save_img_as_buffer = True
        self.output_buffer = None

        self.reset()

    def load_map(self, path) -> np.array:
        return np.genfromtxt(join(path), dtype = int, delimiter = ',')

    def reset(self):
        self.gen_coins()
        self.gen_pacman()
        self.gen_ghosts()
        self.update_map()

    def gen_coins(self):
        random_matrix = np.random.rand(*self.map.shape)
        self.map = np.where(((self.map == Blocks.EMPTY.id) & (random_matrix > 0.2)), Blocks.COIN.id, self.map)
        init_coins_position = np.where(self.map == Blocks.COIN.id)
        for pos in zip(*init_coins_position):
            self.coins.append(
                Coin(position=np.array(pos), state = Blocks.COIN.id)
            )
        self.max_coins = len(self.coins)

    def gen_pacman(self):
        init_pacman_position = np.where(self.map == Blocks.PACMAN.id)
        for pos in zip(*init_pacman_position): #only one iteration
            self.pacman = Pacman(pos, self.map)
        #     self.pacman = Ghost(start_pos=np.array(pos), map=self.map)


    def gen_ghosts(self):
        init_ghosts_position = np.where(self.map == Blocks.GHOST.id)
        for pos in zip(*init_ghosts_position):
            self.ghosts.append(
                Ghost(start_pos=np.array(pos), map=self.map)
            )
        
    def is_done(self):
        return self.check_steps() or self.check_collision() or self.check_coins()

    def check_steps(self):
        return self.max_steps == self.current_steps

    def check_collision(self):
        pacman_position = self.pacman.current_pos
        for ghost in self.ghosts:
            if (pacman_position[0] == ghost.current_pos[0]) and \
               (pacman_position[1] == ghost.current_pos[1]):
                return True
        return False

    def check_coins(self): #if win
        coins = len(np.where(self.map == Blocks.COIN.id)[0])
        
        return coins == 0

    def step(self):
        # überprüfen ob zuerst Pacman oder die Geister ihren Step machen
        self.current_steps += 1

        for ghost in self.ghosts:
            ghost.move()

        self.pacman.predict_new_move()

        reward = self.update_map()

        done = self.is_done()
        return reward, self.map, done

    
    def update_map(self):

        # reset all old positions initially
        self.map = np.where((self.map == Blocks.GHOST.id), Blocks.EMPTY.id, self.map)
        self.map = np.where((self.map == Blocks.PACMAN.id), Blocks.EMPTY.id, self.map)

        # ========Coins========
        # neue position setzen
        for coin in self.coins:
            current_pos = coin.position
            self.map[current_pos[0], current_pos[1]] = Blocks.COIN.id

        # ========Ghosts========
        # neue position setzen
        for ghost in self.ghosts:
            current_pos = ghost.current_pos
            self.map[current_pos[0], current_pos[1]] = Blocks.GHOST.id
    
        # ========Pacman=======
        current_pos = self.pacman.current_pos
        field_id = self.get_field_id(current_pos, change_state = True)
        self.map[current_pos[0], current_pos[1]] = Blocks.PACMAN.id
        return field_id

    def get_field_id(self, current_pos, change_state = False):
        for coin in self.coins:
            if (current_pos[0] == coin.position[0]) and \
               (current_pos[1] == coin.position[1]):
                if change_state:
                   self.coins.remove(coin)
                return Blocks.COIN.id
        return Blocks.EMPTY.id

    def get_collected_points(self) -> int:
        return self.max_coins - len(self.coins)

    def show(self) -> None:
        points_patch = mpatches.Patch(color=Blocks.PACMAN.color, label=f'Pacman Points: {self.get_collected_points()}')
        steps_patch = mpatches.Patch(color=Blocks.PACMAN.color, label=f'Pacman Steps: {self.current_steps}')

        plt.figure(1)
        plt.clf()
        plt.imshow(self.map, cmap=self.cmap)
        plt.legend(handles=[points_patch, steps_patch])
        plt.pause(1e-5)

        if self.save_img_as_buffer:
            with io.BytesIO() as output:
                plt.matshow(self.map, cmap=self.cmap)
                plt.legend(handles=[points_patch, steps_patch])
                plt.savefig(output, format='png')
                plt.close()
                self.output_buffer = output.getvalue()