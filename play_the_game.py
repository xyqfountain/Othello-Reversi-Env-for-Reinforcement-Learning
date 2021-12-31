from reversi_env import Reversi
import cv2
from agant import Agent_PG, Agent_DQN
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    HUMAN_vs_MACHINE = True  # Human are always Black players
    env = Reversi(human_VS_machine=HUMAN_vs_MACHINE)
    # machine = Agent_PG("White", device=device).to(device)
    # machine.load_model(name="brain_PG")
    machine = Agent_DQN("White", device=device).to(device)
    machine.load_model("Brain_DQN_prioritized_White20000")
    if HUMAN_vs_MACHINE:  # human vs machine
        for ep in range(1):
            obs, info = env.reset()  # {"next_player": self.next_player, "next_possible_actions":
            env.render()  # show the initialization
            while True:
                if info["next_player"] == "White":  # machine's turn
                    action = machine.choose_action(obs, info["next_possible_actions"])
                else:  # human's turn
                    action = env.get_human_action()
                obs, _, done, info = env.step(action)
                env.render()
                if done:
                    break
            cv2.waitKey(3000)  # wait for 3 seconds after the end of each ep
        cv2.waitKey()
        env.close()
    else:
        win_counter_white = 0
        round_max = 200
        RENDER = False
        for ep in range(round_max):
            obs, info = env.reset()  # {"next_player": self.next_player, "next_possible_actions":
            if RENDER: env.render()  # show the initialization
            while True:
                if info["next_player"] == "White":  # machine's turn
                    action = machine.choose_action(obs, info["next_possible_actions"])
                else:
                    action = env.get_random_action()
                obs, _, done, info = env.step(action)
                if RENDER: env.render()
                if done:
                    win_counter_white += 1 if info["winner"] == "White" else 0
                    print("Round: {:d}/{:d}, winner is ".format(ep, round_max), info["winner"])
                    break
            # cv2.waitKey()
        print("Winning rate of the white player is {:.2%}".format(win_counter_white/round_max))
        # cv2.waitKey()
        env.close()

