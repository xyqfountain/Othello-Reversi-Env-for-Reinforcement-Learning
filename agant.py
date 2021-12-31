import torch
import torch.nn as nn
import numpy as np
import os
import sys
import random

# np.random.seed(7)
# random.seed(7)

class Brain(nn.Module):
    def __init__(self, a_dim=64, s_dim=64):
        super(Brain, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, a_dim),
            nn.LeakyReLU(),
            # nn.Dropout(0.2),
        )

    def forward(self, x):
        """
        :param x: tensor, [bs, 64]
        :param mask: tensor, torch.bool, [bs, 64]
        :return:
        """
        x = self.net(x)
        return x


class Agent_PG(nn.Module):
    def __init__(self, player="Black", device=torch.device("cuda")):
        super(Agent_PG, self).__init__()
        assert player in ("Black", "White"), "Wrong player color"
        self.id = player
        self.a_dim = 64
        self.s_dim = 64
        self.device = device
        self.brain = Brain(a_dim=self.a_dim, s_dim=self.s_dim).to(device)
        self.ep_obs, self.ep_as, self.ep_rs, self.ep_a_cand = [], [], [], []
        self.gamma = 0.95  # reward decay rate
        self.alpha = 0.2  # soft copy weights from white to black, alpha updates while (1-alpha) remains
        if self.id == "White":
            self.opt = torch.optim.Adam(self.brain.parameters(), lr=1e-4, weight_decay=0.01)
            self.critic = nn.CrossEntropyLoss(reduction="none")  # do not apply mean

    def choose_action(self, obs, a_candicates):
        """
        :param obs: list[list], [8, 8]
        :param a_candicates: a set of tuples
        :return: a tuple of (row, col)
        """
        self.brain.eval()
        obs = torch.tensor(np.array(obs).ravel(), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mask = torch.tensor([[True]*64], dtype=torch.bool).to(self.device)  # shape = [1, 64]
            for r, c in a_candicates:
                mask[0][r * 8 + c] = False   # do not mask a possible action

            probs = self.brain(obs)
            probs = probs.masked_fill(mask, -1e9)
            probs = torch.softmax(probs, dim=1)

            # action = torch.argmax(probs, dim=1).item()  # greedy
            action = np.random.choice(range(64), p=probs.cpu().numpy().ravel())

            action = (action // 8, action % 8)
        return action

    def store_transition(self, obs, a, r, a_candicates):
        """
        :param obs:list[list], 8x8
        :param a: tuple, (row, col)
        :param r: int
        :param a_candicates: a set of actions (row, col)
        :return:
        """
        self.ep_obs.append(np.array(obs).ravel())
        self.ep_as.append(a[0] * 8 + a[1])
        self.ep_rs.append(r)
        self.ep_a_cand.append(a_candicates)

    def learn(self):
        if self.id == "White":
            # discount and normalize episode reward
            discounted_ep_rs_norm = torch.tensor(self.__discount_and_norm_rewards()).to(self.device)
            self.brain.train()
            self.opt.zero_grad()
            obs = torch.tensor(np.vstack(self.ep_obs), dtype=torch.float32).to(self.device)
            labels = torch.tensor(np.array(self.ep_as), dtype=torch.long).to(self.device)
            masks = torch.tensor([[True]*64 for _ in range(len(self.ep_a_cand))]).to(self.device)
            for i, a_cand in enumerate(self.ep_a_cand):
                for r, c in a_cand:
                    masks[i][r * 8 + c] = False

            net_out = self.brain(obs)
            net_out = net_out.masked_fill(masks, -1e9)

            # log_softmax = nn.LogSoftmax(dim=1)(net_out)
            # neg_log_likelihood = nn.NLLLoss(reduction="none")(log_softmax, labels)
            # loss = torch.mul(neg_log_likelihood, discounted_ep_rs_norm)
            # loss = loss.mean()
            loss = torch.mean(torch.mul(self.critic(net_out, labels), discounted_ep_rs_norm))

            loss.backward()
            self.opt.step()
            self.ep_obs, self.ep_as, self.ep_rs, self.ep_a_cand = [], [], [], []
            return loss.item()

    def __discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs, dtype=np.float64)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def weights_assign(self, another:Brain):
        for tgt_param, src_param in zip(self.brain.parameters(), another.parameters()):
            tgt_param.data.copy_(self.alpha * src_param.data + (1. - self.alpha) * tgt_param.data)

    def save_model(self, name:str):
        torch.save(self.brain.state_dict(), os.path.join("models.", name))

    def load_model(self, name="agent_PG"):
        if not os.path.exists(os.path.join("models", name)):
            sys.exit("cannot load %s" % name)
        self.brain.load_state_dict(torch.load(os.path.join("models", name)))


class Agent_DQN(nn.Module):
    def __init__(self, player="Black", device=torch.device("cuda")):
        super(Agent_DQN, self).__init__()
        assert player in ("Black", "White"), "Wrong player color"
        self.id = player
        self.double_q = False  # True = DQN_double, False = DQN_nature
        self.prioritized = True  # True = prioritized mempory replay, False = DQN_nature, https://blog.csdn.net/gsww404/article/details/103673852

        self.a_dim = 64
        self.s_dim = 64
        self.device = device
        self.gamma = 0.95  # reward decay rate
        self.alpha1 = 0.1  # soft copy weights from white to black, alpha1 updates while (1-alpha1) remains
        self.alpha2 = 0.1  # soft copy weights from eval net to target net, alpha2 updates while (1-alpha2) remains
        self.epsilon_max = 1.0
        self.batch_size = 64
        self.epsilon_increment = None  # 0.0005
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max
        # total learning step
        self.learn_step_counter = 0  # count how many times the eval net has been updated, used to set a basis for updating the target net
        self.replace_target_iter = 200

        self.brain_evl = Brain(a_dim=self.a_dim, s_dim=self.s_dim).to(device)
        if self.id == "White":  # only while player learns
            self.brain_tgt = Brain(a_dim=self.a_dim, s_dim=self.s_dim).to(device)
            self.memory_size = 10000
            self.memory_counter = 0
            # initialize zero memory [s, a, r, done, s_, a_cand] = 64+1+1+64+64
            if self.prioritized:  # prioritized experience replay
                self.memory = Memory(self.memory_size)
            else:
                self.memory = np.zeros((self.memory_size, self.s_dim + 1 + 1 + 1 + self.s_dim + self.a_dim), dtype=np.float)
            self.opt = torch.optim.Adam(self.brain_evl.parameters(), lr=1e-3, weight_decay=0.1)
            self.critic = nn.MSELoss()

    def choose_action(self, obs, a_possible):
        """
        :param obs: list[list], shape=[8, 8]
        :param a_possible: a set of tuples (row, col)
        :return: a tuple of (row, col)
        """
        self.brain_evl.eval()
        with torch.no_grad():
            obs = torch.tensor(np.array(obs).ravel(), dtype=torch.float32).unsqueeze(0).to(self.device)  # shape = (1, 64)
            mask = torch.tensor([[True] * 64], dtype=torch.bool).to(self.device)  # shape = (1, 64)
            for r, c in a_possible:
                mask[0][r * 8 + c] = False  # do not mask a possible action

            probs = self.brain_evl(obs)  # (1, 64)
            probs = probs.masked_fill(mask, -1e9)
            probs = torch.softmax(probs, dim=1)  # all masked prob equal to 0 after this step

            # action = torch.argmax(probs, dim=1).item()  # greedy
            # action = np.random.choice(range(64), p=probs.cpu().numpy().ravel())  # p-distribution
            # e-greedy
            if np.random.uniform() < self.epsilon:
                action = torch.argmax(probs, dim=1).item()
                action = (action // 8, action % 8)
            else:
                action = random.choice(list(a_possible))
        return action

    def store_transition(self, obs, a, r, done, obs_, a_possible):
        """
        :param obs， obs_:list[list], 8x8
        :param a: tuple, (row, col)
        :param r: int
        :param done: bool
        :param a_candicates: a set of actions (row, col), it wont be used in the learning process so it does not matter if stored
        :return:
        """
        if self.id == "White":
            a_mask = np.ones(self.a_dim)
            for row, col in a_possible:
                a_mask[row * 8 + col] = 0  # do not mask a possible action
            transition = np.hstack((np.array(obs).ravel(), a[0]*8+a[1], r, done, np.array(obs_).ravel(), a_mask))  # (64+1+1+1+64+64)

            if self.prioritized:  # prioritized experience replay
                self.memory.store(transition)
            else:
                index = self.memory_counter % self.memory_size
                self.memory[index] = transition
                self.memory_counter += 1
                if self.memory_counter == self.memory_size*3:  # avoid too large number
                    self.memory_counter -= self.memory_size

    def weights_assign(self, another: Brain):
        """
        accept weights of the brain_eval from the white player
        :param another:
        :return:
        """
        if self.id == "Black":
            with torch.no_grad():
                for tgt_param, src_param in zip(self.brain_evl.parameters(), another.parameters()):
                    tgt_param.data.copy_(self.alpha1 * src_param.data + (1.0 - self.alpha1) * tgt_param.data)

    def __tgt_evl_sync(self):
        """
        this function is to assign the weight of the eval network to the target network
        :return:
        """
        if self.id == "White":
            for tgt_param, src_param in zip(self.brain_tgt.parameters(), self.brain_evl.parameters()):
                tgt_param.data.copy_(self.alpha2 * src_param.data + (1.0 - self.alpha2) * tgt_param.data)

    def learn(self):
        if self.id == "White":  # only white player learns
            self.brain_evl.train()
            self.brain_tgt.eval()  # do not train it.
            self.opt.zero_grad()
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.__tgt_evl_sync()

            if self.prioritized:
                tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
                ISWeights = torch.tensor(ISWeights, dtype=torch.float).squeeze().to(self.device)
            else:
                if self.memory_counter > self.memory_size:
                    sample_index = np.random.choice(self.memory_size, size=self.batch_size)
                else:
                    sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
                batch_memory = self.memory[sample_index]

            obs = torch.tensor(batch_memory[:, :self.s_dim], dtype=torch.float).to(self.device)
            a = torch.tensor(batch_memory[:, self.s_dim], dtype=torch.long).to(self.device)
            r = torch.tensor(batch_memory[:, self.s_dim+1], dtype=torch.float).to(self.device)
            done = torch.tensor(batch_memory[:, self.s_dim+2], dtype=torch.bool).to(self.device)
            obs_ = torch.tensor(batch_memory[:, -self.s_dim-self.a_dim: -self.a_dim], dtype=torch.float).to(self.device)
            # a_possible = batch_memory[:, -self.a_dim:]  # a_possbile is not used

            q_eval = self.brain_evl(obs)  # tensor, shape = [bs, 64]
            q_eval_wrt_a = torch.gather(q_eval, dim=1, index=a.view(-1, 1)).squeeze()  # [bs, ]
            with torch.no_grad():  # refer to https://pytorch.org/docs/stable/autograd.html#torch.autograd.set_grad_enabled
                q_next = self.brain_tgt(obs_)  # tensor, shape = [bs, 64]
                if self.double_q:  # double DQN
                    q_eval4next = self.brain_evl(obs_)  # [bs, 64]
                    max_act4next = torch.argmax(q_eval4next, dim=1)  # [bs, ]
                    slected_q_next = torch.gather(q_next, dim=1, index=max_act4next.view(-1, 1)).squeeze() # [bs, ]
                    q_target = r + self.gamma * slected_q_next  # [bs, ]
                else: # nature DQN
                    q_target = r + self.gamma * torch.max(q_next, dim=1)[0]  # [bs, ]
                q_target[done] = r[done]  # [bs, ]

            if self.prioritized:
                with torch.no_grad():
                    abs_errors = torch.abs(q_target - q_eval_wrt_a).cpu().data.numpy()
                loss = torch.mean(ISWeights * torch.square(q_target - q_eval_wrt_a))
                self.memory.batch_update(tree_idx, abs_errors)
            else:
                loss = self.critic(q_target, q_eval_wrt_a)
                # loss = torch.mean(torch.square(q_target - q_eval_wrt_a))
            loss.backward()
            self.opt.step()
            self.learn_step_counter += 1
            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
            return loss.item()

    def reward_transition_update(self, reward:float):
        """
        if it is the Black that take the last turn, the reward the white player obtained should be updated because the winner has been determined
        :param reward: float
        :return:
        """
        if self.id == "White":
            if self.prioritized:
                index = (self.memory.tree.data_pointer - 1) % self.memory_size
                self.memory.tree.data[index][self.s_dim+1] = reward
            else:
                index = (self.memory_counter - 1) % self.memory_size
                self.memory[index, self.s_dim+1] = reward

    def save_model(self, name:str):
        torch.save(self.brain_evl.state_dict(), os.path.join("models.", name))

    def load_model(self, name="brain_DQN_nature"):
        if not os.path.exists(os.path.join("models", name)):
            sys.exit("cannot load %s" % name)
        self.brain_evl.load_state_dict(torch.load(os.path.join("models", name)))


class SumTree:
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, done, s_, a_possible) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])  # maximal p in data
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new transition

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        if min_prob == 0:
            min_prob = 0.00001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
