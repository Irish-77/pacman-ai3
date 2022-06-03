import io
import numpy as np
import matplotlib.colors as mat_colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from os.path import join

from ghost import Ghost
from coin import Coin
from pacman import Pacman
from options import Blocks, Movements

class Environment():
    """Environment that manages all objects and the map
    """

    def __init__(self,
                path:str='map.csv') -> None:

        """Constructor for the environment

        The environment manages all the coins and ghosts which is why the
        environment is considered as the center of the application, all players
        interact with each other through the environment.

        The map was developed using Excel/Google Sheets. Each cell was filled
        with a certain value. Through conditional formatting, the color of the
        cell changed afterwards, so you could always see what Excel looked like
        in real time. The Excel was then exported in CSV format. 

        Args:
            path (str, optional): Path to the map. Defaults to 'map.csv'.
        """
        
        # Initializing objects
        self.init_map = self.load_map(path)
        self.map = None
        self.ghosts = []
        self.coins = []
        self.pacman = None

        # Initializing game related measures
        self.max_coins = 0
        self.max_steps = 1000
        self.current_steps = 0
        
        # Initializing visuals for the GUI
        colors = [x.color for x in Blocks]
        self.cmap = mat_colors.ListedColormap(colors)

        # Initializing empty output buffer to store the image later
        self.output_buffer = None

        # Creating all necessary objects
        self.reset()

    def load_map(self, path:str) -> np.array:
        """Loads the map from a CSV-file and parses it to a numy matrix

        Args:
            path (str): Path to the map
        Returns:
            np.array: Map as numpy matrix
        """
        return np.genfromtxt(join(path), dtype = int, delimiter = ',')

    def reset(self) -> np.array:
        """Resets the entire environment

        In order to simulate a new epsiode the environment must be set to
        default. The initial map is reloaded, coins are generated on the playing
        field, Pacman and the ghosts are initialized. 

        Returns:
            np.array: Current map
        """
        self.map = self.init_map
        self.current_steps = 0
        self.gen_coins()
        self.gen_pacman()
        self.gen_ghosts()
        self.update_map()
        return self.map

    def gen_coins(self) -> None:
        """Generates coins on the map

        The coins are randomly positioned around the map. On the positions of
        Pacman, the ghosts and on the walls no coins are positioned. In total,
        only 80% of the available spaces will have a coin placed on them.

        Furthermore, all coins will be appended to the coin list. The maximum
        amount of collectalbe coins corresponds to the initial length of the
        coin list. 
        """

        random_matrix = np.random.rand(*self.map.shape)
        self.map = np.where(((self.map == Blocks.EMPTY.id) & \
                            (random_matrix > 0.2)),
                            Blocks.COIN.id, self.map)
        init_coins_position = np.where(self.map == Blocks.COIN.id)
        self.coins = []
        for pos in zip(*init_coins_position):
            self.coins.append(
                Coin(position=np.array(pos))
            )
        self.max_coins = len(self.coins)

    def gen_pacman(self) -> None:
        """Generates/initilaises Pacman

        The Pacman object is created, the initial position is passed to the
        object. Since the position array cotains only one location, the for loop
        will only be iterated once.

        """
        init_pacman_position = np.where(self.map == Blocks.PACMAN.id)
        for pos in zip(*init_pacman_position): #only one iteration
            self.pacman = Pacman(pos, self.map)

    def gen_ghosts(self) -> None:
        """Generates ghosts

        First, the spawn points of the ghosts pre-defined in the map are read.
        For each of these spawn points a ghost is created. All ghosts are added
        to a list that allows the environment to easily communicate with all
        ghosts.
        """

        self.ghosts = []
        init_ghosts_position = np.where(self.map == Blocks.GHOST.id)
        for pos in zip(*init_ghosts_position):
            self.ghosts.append(
                Ghost(start_pos=np.array(pos), map=self.map)
            )
        
    def is_done(self) -> bool:
        """Checks if current simulation is done

        The current simulation is done when one of the following three
        conditions are met (true):
        1. The maximum number of defined steps was exceeded,
        2. Pacman has collided with a ghost, or
        3. all coins have been collected by Pacman.

        Returns:
            bool: true if current simulation is done
        """
        return  self.check_steps() or \
                self.check_collision() or \
                self.check_coins()

    def check_steps(self) -> bool:
        """Checks if maximum number of defined steps was exceeded 

        Returns:
            bool: true if max. number of steps was exceeded
        """
        return self.max_steps == self.current_steps

    def check_collision(self) -> bool:
        """Checks if Pacman has collided with any ghosts

        Returns:
            bool: true if Pacman hit another ghost
        """
        pacman_position = self.pacman.current_pos
        for ghost in self.ghosts:
            if (pacman_position[0] == ghost.current_pos[0]) and \
               (pacman_position[1] == ghost.current_pos[1]):
                return True
        return False

    def check_coins(self) -> bool:
        """Checks if all coins have been collected by Pacman

        Returns:
            bool: true if length of coin list is empty 🡢 no coins left
        """
        #if win
        coins = len(np.where(self.map == Blocks.COIN.id)[0])
        
        return coins == 0

    def step(self,
            direction:Movements
            ) -> tuple[int, np.array, bool]:
        """Performs a new iteration within current simulation

        During the step method, the ghosts and Pacman are set to the new
        position. In addition, a check is made to see if the simulation is
        complete. Furthermore, the map is updated afterwards.

        Args:
            direction (Movements): Direction in that Pacman should move

        Returns:
            Tuple[int, np.array, bool]:
                [0]: reward (0: no coin collected, 1: coin collected)
                [1]: current state of the map
                [2]: true if simulation is done
        """
        self.current_steps += 1

        self.pacman.move(direction)
        done = self.check_collision()
        if not done:
            for ghost in self.ghosts:
                ghost.move()
            done = self.is_done()

        reward = self.update_map()

        return reward, self.map, done

    
    def update_map(self) -> int:
        """Updates the map after each iteration

        This method updates all positions of all mutable objects on the map. To
        do this, the old positions are first overwritten and the fields are set
        to empty. Then all objects are redrawn on the map. In addition, it is
        checked whether Pacman is now set to Coin. If this is the case, 1 is
        returned as reward. The field of the coin that Pacman has picked up does
        not need to be reseted on the map, because Pacman is already on the same
        field as well, accordingly this position is reseted too.  

        Returns:
            int: reward for Pacmans new position
        """

        # resets all old positions to default empty fields
        # 🡢 will be overwritten
        self.map = np.where((self.map == Blocks.GHOST.id),
                            Blocks.EMPTY.id,
                            self.map)
        self.map = np.where((self.map == Blocks.PACMAN.id),
                            Blocks.EMPTY.id,
                            self.map)

        # ========Coins========
        # set new postion
        for coin in self.coins:
            current_pos = coin.position
            self.map[current_pos[0], current_pos[1]] = Blocks.COIN.id

        # ========Ghosts========
        # set new postion
        for ghost in self.ghosts:
            current_pos = ghost.current_pos
            self.map[current_pos[0], current_pos[1]] = Blocks.GHOST.id
    
        # ========Pacman=======
        # set new postion
        current_pos = self.pacman.current_pos
        field_id = self.get_field_id(current_pos, change_state = True)
        self.map[current_pos[0], current_pos[1]] = Blocks.PACMAN.id

        return field_id

    def get_field_id(self,
                    current_pos:np.array,
                    change_state:bool = False) -> int:
        """Returns the value of a current position on the map

        After each new move of Pacman it is checked if Pacman is on a field
        where there is a coin located. If this is the case and the flag variable
        ´change_state´ is set to true, the coin is picked up by Pacman, which
        corresponds to removing the coin from the list. In addition, 1 is
        returned, which indicates a positive outcome. Otherwise 0 is returned,
        which means that no coin was collected.

        Args:
            current_pos (np.array):
                Position to be checked
            change_state (bool, optional):
                If true the coin will be removed from the coin list, so that
                Pacman can collect the reward only once. Defaults to False.

        Returns:
            int: value of the field (0: no coin collected, 1: coin collected)
        """

        for coin in self.coins:
            if (current_pos[0] == coin.position[0]) and \
               (current_pos[1] == coin.position[1]):
                if change_state:
                   self.coins.remove(coin)
                return 1
        return 0

    def get_collected_points(self) -> int:
        """Returns the number of collected points by Pacman

        The coins/points collected so far correspond to the difference between
        the initial stock of coins and the current amount. The amount can be
        read from the length of the list in which all coins are stored. 

        Returns:
            int: amount of collected points
        """
        return self.max_coins - len(self.coins)

    def get_patches(self) -> tuple[mpatches.Patch]:
        """Returns patches that are displayed within the GUI
        
        Within the patches the current number of steps and the points so far
        achieved by Pacman from the current simulation are stored. These patches
        are displayed in the upper right corner of the GUI.

        Returns:
            tuple[mpatches.Patch]:
                list containing two patches: one for current step (iteration)
                and the second one for the achieved points so far  
        """

        points_patch = mpatches.Patch(color=Blocks.PACMAN.color,
                        label=f'Pacman Points: {self.get_collected_points()}')
        
        steps_patch = mpatches.Patch(color=Blocks.PACMAN.color,
                        label=f'Pacman Steps: {self.current_steps}')

        return points_patch, steps_patch

    def show(self) -> None:
        """Shows current frame of simulation

        The matplotlib figure updates in real time. In addition, the current
        number of steps and points achieved so far are displayed.
        """

        points_patch, steps_patch = self.get_patches()

        plt.figure(1)
        plt.clf()
        plt.imshow(self.map, cmap=self.cmap)
        plt.legend(handles=[points_patch, steps_patch])
        plt.pause(1e-5)
    
    def write_img_to_buffer(self) -> None:
        """Saves current frame to IO-Buffer

        The same frame that is displayed in the ´show´ method will be now stored
        """
        points_patch, steps_patch = self.get_patches()
        with io.BytesIO() as output:
            plt.matshow(self.map, cmap=self.cmap)
            plt.legend(handles=[points_patch, steps_patch])
            plt.savefig(output, format='png')
            plt.close()
            self.output_buffer = output.getvalue()