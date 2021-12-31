from reversi_env import Reversi
from agant import Agent_PG, Agent_DQN
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    model = "DQN"  # "PG", "DQN"
    if model == "PG":
        white_check_point = "agent_White_PG2000"
        black_check_point = "agent_Black_PG2000"
        agent_White = Agent_PG("White", device=device).to(device)
        agent_Black = Agent_PG("Black", device=device).to(device)
        if white_check_point:
            agent_White.load_model(white_check_point)
        if black_check_point:
            agent_Black.load_model(black_check_point)

        env = Reversi(human_VS_machine=False)
        reward_history, winning_rate = [], []
        is_White = []
        max_epoch = 10000
        RENDER = False
        for ep in range(1, max_epoch+1):
            ep_reward = []
            obs, info = env.reset()
            done = False
            if RENDER: env.render()
            while True:
                next_palyer = info["next_player"]
                next_possible_actions = info["next_possible_actions"]

                if next_palyer == "White":  # We train the white
                    action = agent_White.choose_action(obs, next_possible_actions)
                    obs_, reward, done, info = env.step(action)
                    ep_reward.append(reward)
                    agent_White.store_transition(obs, action, reward, next_possible_actions)
                else:
                    # action = agent_Black.choose_action(obs, next_possible_actions)
                    action = env.get_random_action()
                    obs_, reward, done, info = env.step(action)
                    if done:
                        if info["winner"] == "Black":  # when black take the last turn and game over, rewards of white player should be updated
                            agent_White.ep_rs[-1] -= 10
                            ep_reward[-1] -= 10
                        elif info["winner"] == "White":
                            agent_White.ep_rs[-1] += 10
                            ep_reward[-1] += 10
                        else:  # "Tie"
                            agent_White.ep_rs[-1] += 2
                            ep_reward[-1] += 2
                obs = copy.deepcopy(obs_)

                if RENDER: env.render()
                if done:  # Game Over
                    loss = agent_White.learn()
                    print("ep: {:d}/{:d}, white player taining loss value: {:.4f}".format(ep, max_epoch, loss))
                    is_White.append(True if info["winner"] == "White" else False)
                    break
            reward_history.append(np.sum(ep_reward))

            if ep % 20 == 0:  # update the weights of the black player
                winning_rate.append(np.mean(is_White))
                is_White = []
                print("ep: {:d}/{:d}, white player winning rate in latest 20 rounds: {:.2%}.".format(ep, max_epoch, winning_rate[-1]))

            if len(winning_rate) >= 3 and all([w >= 0.65 for w in winning_rate[-3:]]):
                agent_Black.weights_assign(agent_White.brain)
                print("ep: {:d}/{:d}, black player updated.".format(ep, max_epoch))

        # end of training
        agent_White.save_model("agent_PG")
        # plot
        plt.figure("White winning rate")
        plt.plot(range(0, max_epoch, 20), winning_rate)
        plt.show()
    elif model == "DQN":
        white_check_point = None
        black_check_point = None
        agent_White = Agent_DQN("White", device=device).to(device)
        agent_Black = Agent_DQN("Black", device=device).to(device)
        if white_check_point:
            agent_White.load_model(white_check_point)
        if black_check_point:
            agent_Black.load_model(black_check_point)

        env = Reversi(human_VS_machine=False)
        reward_history, winning_rate = [], []
        best_model, best_winning_rate = None, 0.  # the one obtained the highest winning rate, regardless of opponent
        is_White = []
        max_epoch = 20000
        dominant_counter_white = 0
        RENDER = False
        for ep in range(1, max_epoch + 1):
            ep_reward = []
            obs, info = env.reset()
            done = False
            if RENDER: env.render()
            while True:
                next_palyer = info["next_player"]
                next_possible_actions = info["next_possible_actions"]

                if next_palyer == "White":  # We train the white
                    action = agent_White.choose_action(obs, next_possible_actions)
                    obs_, reward, done, info = env.step(action)
                    ep_reward.append(reward)
                    agent_White.store_transition(obs, action, reward, done, obs_, next_possible_actions)
                else:
                    action = agent_Black.choose_action(obs, next_possible_actions)
                    # action = env.get_random_action()
                    obs_, reward, done, info = env.step(action)
                    if done:
                        if info["winner"] == "Black":  # when black take the last turn and game over, rewards of white player should be updated
                            agent_White.reward_transition_update(-10.)
                        elif info["winner"] == "White":
                            agent_White.reward_transition_update(10.)
                        else:  # "Tie"
                            agent_White.reward_transition_update(2.)
                obs = copy.deepcopy(obs_)

                if RENDER: env.render()
                if done:  # Game Over
                    loss = agent_White.learn()
                    print("ep: {:d}/{:d}, white player taining loss value: {:.4f}".format(ep, max_epoch, loss))
                    is_White.append(True if info["winner"] == "White" else False)
                    break
            reward_history.append(np.sum(ep_reward))

            if ep % 20 == 0:  # log winning rate in every 20 eps
                winning_rate.append(np.mean(is_White))
                is_White = []
                print("ep: {:d}/{:d}, white player winning rate in latest 20 rounds: {:.2%}.".format(ep, max_epoch, winning_rate[-1]))
                if best_winning_rate <= winning_rate[-1]:
                    best_model = copy.deepcopy(agent_White)
                    best_winning_rate = winning_rate[-1]
                if winning_rate[-1] >= 0.60:
                    dominant_counter_white += 1
                else:
                    dominant_counter_white = 0
                if dominant_counter_white >= 3:
                    dominant_counter_white = 0
                    agent_Black.weights_assign(agent_White.brain_evl)
                    print("ep: {:d}/{:d}, black player updated.".format(ep, max_epoch))

        # end of training
        agent_White.save_model("Brain_DQN_prioritized_White20000")
        agent_Black.save_model("Brain_DQN_prioritized_Black20000")
        best_model.save_model("Brain_DQN_prioritized_Best20000")
        # plot
        plt.figure("White winning rate")
        plt.plot(range(0, max_epoch, 20), winning_rate)
        plt.show()