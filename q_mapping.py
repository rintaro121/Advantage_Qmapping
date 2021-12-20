import math
import numpy as np
import torch

from log import Log
from PPO import PPO

state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std \
        = 13, 4, 0.0003, 0.001, 0.99, 80, 0.2, False, 0.6

CHs = ["CH1", "CH2", "CH3", "CH4"]
action_order = ["LEFT", "UP", "RIGHT", "DOWN"]

MAX_ENTROPY_VALUE = 1.3863

class Qmapping:
    def __init__(self, path, args):
        self.operation_history = {}
        self.q_mapping = {}
        self.PPO = self.getPPO(path)
        self.actor = self.PPO.policy.actor
        self.args = args

        self.log = Log()
    
        for ch in CHs:
            self.operation_history[ch] = 0
            self.q_mapping[ch] = [0.25, 0.25, 0.25, 0.25]  # 左，上，右，下の確率


    def interpret(self, operation_ch, obs, calibration = False, frame=0):
        self.operation_history[operation_ch] += 1
        #alpha = self.calc_alpha(self.operation_history[operation_ch], 0.05, 200)  # 0.5 ~ 1
        if calibration:
            alpha = self.calc_beta(frame, 0.05, 50)
            #alpha = min(0.8, alpha)
        else:
            alpha = self.calc_beta(frame, 0.05, 100)    #   0 ~ 1
            #alpha = 0

        self.log.beta_log[operation_ch].append(alpha)

        #q = self.Q(torch.tensor(obs, dtype=torch.float32))
        with torch.no_grad():
            advantage_value = self.calc_advantage(obs)

        if self.args.dir == 'vertical':
            self.log.vertical_q_value_log["LEFT"].append(advantage_value[0])
            self.log.vertical_q_value_log["UP"].append(advantage_value[1])
            self.log.vertical_q_value_log["RIGHT"].append(advantage_value[2])
            self.log.vertical_q_value_log["DOWN"].append(advantage_value[3])
        elif self.args.dir == 'diagonal':
            self.log.diagonal_q_value_log["RIGHT_DOWN"].append(advantage_value[0])
            self.log.diagonal_q_value_log["LEFT_DOWN"].append(advantage_value[1])
            self.log.diagonal_q_value_log["RIGHT_UP"].append(advantage_value[2])
            self.log.diagonal_q_value_log["LEFT_UP"].append(advantage_value[3])
            
        advantage_value = advantage_value.detach().numpy()
        #softmax_advantage_value = self.softmax(advantage_value)
        
        self.log.coef.append(alpha)

        # ∀a,Q_m(o_t,a) ← (1 - α)・Q_m(o_t,a) + α・Q(s_t,q)
        entropy = self.PPO.calc_entropy(obs)
        entropy = entropy.detach().numpy()
        coef = 1 - entropy / MAX_ENTROPY_VALUE  
        
        # 押されたCHについてのQ-maappingを更新していく
        for action_idx in range(4):
            # 0:左，1:右，2:上，3:下
            
            tmp_ch = "CH" + str(action_idx+1)

            Qm_value = self.q_mapping[operation_ch][action_idx]
            
            #new_Qm_value = (1 - alpha) * (1 - coef) * Qm_value + alpha * coef * advantage_value [action_idx]
            new_Qm_value = (1-alpha) * Qm_value + alpha * advantage_value[action_idx]

            # diff_qmapping = abs(new_Qm_value - Qm_value)
            # self.log.qmapping_diff_log[tmp_ch].append(diff_qmapping)

            self.q_mapping[operation_ch][action_idx] = new_Qm_value
        
            # self.log.q_mapping_value_log[tmp_ch].append((1 - alpha) * Qm_value + alpha * q[action_idx])
            # self.log.scaled_q_mapping_value_log[tmp_ch].append(self.q_mapping[operation_ch][action_idx])

        ret_action_index = np.argmax(self.q_mapping[operation_ch])

        return ret_action_index
    


    def reversable_interpret(self, operation_ch, obs, calibration = False, frame=0, play_only=True):
        self.operation_history[operation_ch] += 1
        if play_only:
            alpha = alpha = self.calc_beta(frame, 0.05, 300)
        else:
            if calibration:
                alpha = self.calc_beta(frame, 0.05, 20)
            else:
                # alpha = self.calc_beta(frame, 0.05, 100) / 2
                alpha = 0
                
        self.log.beta_log[operation_ch].append(alpha)

        with torch.no_grad():
            advantage_value = self.calc_advantage(obs)

        if self.args.dir == 'vertical':
            self.log.vertical_q_value_log["LEFT"].append(advantage_value[0])
            self.log.vertical_q_value_log["UP"].append(advantage_value[1])
            self.log.vertical_q_value_log["RIGHT"].append(advantage_value[2])
            self.log.vertical_q_value_log["DOWN"].append(advantage_value[3])
        elif self.args.dir == 'diagonal':
            self.log.diagonal_q_value_log["RIGHT_DOWN"].append(advantage_value[0])
            self.log.diagonal_q_value_log["LEFT_DOWN"].append(advantage_value[1])
            self.log.diagonal_q_value_log["RIGHT_UP"].append(advantage_value[2])
            self.log.diagonal_q_value_log["LEFT_UP"].append(advantage_value[3])
            
        advantage_value = advantage_value.detach().numpy()
        
        self.log.coef.append(alpha)

        # ∀a,Q_m(o_t,a) ← (1 - α)・Q_m(o_t,a) + α・Q(s_t,q)
        entropy = self.PPO.calc_entropy(obs)
        entropy = entropy.detach().numpy()
        coef = 1 - entropy / MAX_ENTROPY_VALUE  
        
        # 押されたCHについてのQ-maappingを更新していく
        max_Qm_value = 0
        for action_idx in range(4):

            Qm_value = self.q_mapping[operation_ch][action_idx]
            new_Qm_value = (1 - alpha) * Qm_value + alpha * advantage_value[action_idx]
            #new_Qm_value = (1 - alpha) * (1-) * Qm_value + alpha * advantage_value[action_idx]

            self.q_mapping[operation_ch][action_idx] = new_Qm_value
        
        ret_action_index = np.argmax(self.q_mapping[operation_ch])

        # 押されたCHと逆のCHのQ-mapping更新
        def get_opposite_ch(ch):
            if ch == "CH1":
                return "CH3"

            elif ch == "CH2":
                return "CH4"

            elif ch == "CH3":
                return "CH1"

            elif ch == "CH4":
                return "CH2"
        
        def get_opposite_act(act):
            if act == 0:
                return 2
            elif act == 1:
                return 3
            elif act == 2:
                return 0
            elif act == 3:
                return 1
            
        opposite_ch = get_opposite_ch(operation_ch)
        opposite_act = get_opposite_act(ret_action_index)

        max_advantage_value = max(advantage_value)

        opposite_Qm_value = self.q_mapping[opposite_ch][opposite_act]

        new_opposite_Qm_value = (1-alpha) * opposite_Qm_value + alpha * max_advantage_value

        self.q_mapping[opposite_ch][opposite_act] = new_opposite_Qm_value

        self.q_mapping[opposite_ch] = self.softmax(self.q_mapping[opposite_ch])

        #softmax()入れる必要あり？
        return ret_action_index        

    def softmax(self,x):
        ret = []
        deno = 0
        for i in x:
            deno += math.exp(i)
        for i in x:
            ret.append(math.exp(i)/deno)

        return ret


    def calc_alpha(self, x, k = 30, x_0 = 0.08):
        return 1 - self.logistic(x,k,x_0)

    def calc_beta(self, x, k = 30, x_0 = 0.08):
        return (1 - 1 / (1 + math.exp(-k * (x - x_0))))

    def calc_coef_from_qvalue(self, q):
        return np.std(q) * 50

    def logistic(self, x, k = 30, x_0 = 0.08):
        return (1 / (1 + math.exp(-k * (x - x_0)))) / 2

    def getPPO(self, q_path, obs_dim = 13, action_dim = 4):
        net = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
        net.load(q_path)
        return net
    
    def calc_advantage(self, obs):
        with torch.no_grad():
            advantage_value = self.actor(torch.tensor(obs, dtype=torch.float32))
        return advantage_value