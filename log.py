class Log:
    def __init__(self):
        self.beta_log = {
            "CH1" : [],
            "CH2" : [],
            "CH3" : [],
            "CH4" : [],
        }
        
        self.vertical_q_value_log = {
            "LEFT"  : [],
            "UP"    : [],
            "RIGHT" : [],
            "DOWN"  : [],
        }

        self.move_cnt = {
            "LEFT"  : 0,
            "UP"    : 0,
            "RIGHT" : 0,
            "DOWN"  : 0,
        }
        
        self.push_cnt = {
            "LEFT"  : 0,
            "UP"    : 0,
            "RIGHT" : 0,
            "DOWN"  : 0,
        }
        
        self.diagonal_q_value_log = {
            "RIGHT_DOWN":[],
            "LEFT_DOWN" : [],
            "RIGHT_UP" : [],
            "LEFT_UP" : []
        }
        self.q_mapping_value_log = {
            "CH1" : [],
            "CH2" : [],
            "CH3" : [],
            "CH4" : [],
        }

        self.scaled_q_mapping_value_log = {
            "CH1" : [],
            "CH2" : [],
            "CH3" : [],
            "CH4" : [],
        }

        self.operation_log = []

        self.coef = []

        self.qmapping_diff_log = {
            "CH1" : [],
            "CH2" : [],
            "CH3" : [],
            "CH4" : [],
        }

        self.puressure_log = []

        self.observation_log = []
    
    def push_cnt_update(self, action_id):
        dir = self.action_id_to_direction(action_id)
        self.push_cnt[dir] += 1

    def move_cnt_update(self, action_num):
        dir = self.action_num_to_direction(action_num)
        self.move_cnt[dir] += 1

    def action_id_to_direction(self, action_id):
        direction = ""

        if action_id == "CH2":
            direction = 'UP'
        elif action_id == "CH3":
            direction = 'RIGHT'
        elif action_id == "CH4":
            direction = 'DOWN'
        elif action_id == "CH1":
            direction = 'LEFT'

        return direction

    def action_num_to_direction(self, action_num):
        direction = ""

        if action_num == 0:
            direction = "LEFT"
        elif action_num == 1:
            direction = "UP"
        elif action_num == 2:
            direction = "RIGHT"
        elif action_num == 3:
            direction = "DOWN"

        return direction
        

    def operation_and_action_log_update(self, action_id, action_num):
        push_dir = self.action_id_to_direction(action_id)
        move_dir = self.action_num_to_direction(action_num)
        self.operation_log.append([push_dir, move_dir])

    def puressure_log_update(self, info):
        self.puressure_log.append(info)

    def observation_log_update(self, obs):
        self.observation_log.append(obs)

