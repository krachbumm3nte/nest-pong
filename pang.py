import sys
import nest
import argparse
import pong
import numpy as np
import pickle
import time
import os
import logging
import gzip
import datetime

from pong_net import POLL_TIME, PongNet

class AIPong:
    """A class to run and store pong simulations of two competing spiking neural networks
    """

    def __init__(self):
        self.game = pong.GameOfPong()
        # competitors are a network with background noise added to its motor neurons, and one without
        self.left_network = PongNet(with_noise=False)
        self.right_network = PongNet(with_noise=True)



    def run_games(self, folder="", max_runs=5000):
        """run a simulation of pong games and store the results

        Args:
            folder (str, optional): output folder for simulation data (performance of both networks and game state at every iteration). Defaults to current timestamp (YYYY-MM-DD-HH-MM-SS).
            max_runs (int, optional): Number of iterations to simulate. Defaults to 5000.
        """
        if folder == "":
            folder = '{0:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now())
        if os.path.exists(folder):
            print(f"output folder {folder} already exists!")
            sys.exit()   
        os.mkdir(folder)


        self.gamestate_history = []

        start_time = time.time()
        l_score, r_score = 0, 0
        self.run = 0

        while self.run < max_runs:
            self.ball_cell = self.game.ball.get_cell()[1]

            if self.run % 100 == 0:
                logging.info(f"{round(time.time() - start_time, 2)}: Run {self.run}, score: {l_score, r_score}, mean rewards: {round(np.mean(self.left_network.mean_reward))}, {round(np.mean(self.right_network.mean_reward))}") 
            
            self.left_network.set_input_spiketrain(self.ball_cell, self.run)
            self.right_network.set_input_spiketrain(self.ball_cell, self.run)

            logging.debug("Running simulation...")
            nest.Simulate(POLL_TIME)

            for network, paddle in zip([self.left_network, self.right_network], [self.game.l_paddle, self.game.r_paddle]):
                #network.set_input_spiketrain(self.ball_cell, self.run)
                network.poll_network()
                network.reward_by_move()
                network.reset()

                position_diff = network.winning_neuron - paddle.get_cell()[1]
                if position_diff > 0:
                    paddle.move_up()
                elif position_diff == 0:
                    paddle.dont_move()
                else:
                    paddle.move_down()
            
            self.game.step()

            self.run += 1

            self.gamestate_history.append((self.game.ball.get_pos(), self.game.l_paddle.get_pos(), self.game.r_paddle.get_pos(), l_score, r_score))
            

            if self.game.result == pong.RIGHT_SCORE:
                self.game.reset_ball(False)
                r_score += 1
            elif self.game.result == pong.LEFT_SCORE:
                self.game.reset_ball(True)
                l_score += 1
        
        end_time = time.time()
        logging.info(f"simulation of {max_runs} runs complete after: {datetime.timedelta(seconds=end_time-start_time)}")


        logging.info("saving game data...")
        with open(os.path.join(folder, "gamestate.pkl"), "wb") as f:
            pickle.dump(self.gamestate_history, f)


        logging.info("saving performance data...")
        with gzip.open(os.path.join(folder, f"data_right.pkl.gz"), "w") as file:
            pickle.dump(self.right_network.get_performance_data(), file)

        with gzip.open(os.path.join(folder, f"data_left.pkl.gz"), "w") as file:
            pickle.dump(self.left_network.get_performance_data(), file)

        logging.info("done.")


if __name__ == "__main__":
    nest.set_verbosity("M_WARNING")

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",
                        type=int,
                        default=20000,
                        help="Number of runs to perform.")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Verbose debugging output.")
    parser.add_argument("--folder",
                        type=str,
                        default="",
                        help="Folder to save experiments to.")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    aipong = AIPong()
    aipong.run_games(max_runs=args.runs, folder=args.folder)
