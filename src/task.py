import numpy as np
from my_environments import BasicEnv

from stable_baselines3 import PPO


class Task():

    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.obs = env.reset()
        self.env.IsRender = True

    def move_to(self, pos):
        self.env.set_waypoint(pos)
        while True:
            try:
                action, _ = self.model.predict(self.obs)
                self.obs, reward, done, info = self.env.step(action)
            except Exception as e:
                print(f"Error during prediction or step: {e}")
                return False

            if done:
                if info['state'] == "REACH":
                    return True
                else:
                    return False

    def move(self):
        start_pos = self.env.get_waypoint()
        end_pos = self.env.get_waypoint()
        # end_pos = start_pos + np.array([0.0, -0.1, -0.1])
        # end_pos = np.array([-0.0071927, 0.09211658, 0.48778505])
        waypoints = self.generate_waypoints(start_pos, end_pos)
        cnt = 0
        for waypoint in waypoints:
            flag = self.move_to(waypoint)
            if flag:
                cnt += 1
                continue
            else:
                print(f"未到达目标点: {waypoint}, {cnt}/{len(waypoints)}")
                return False
        print(f"到达目标点: {waypoint}, {cnt}/{len(waypoints)}")
        return True

    def generate_waypoints(self, start_pos, end_pos, num_steps=3):
        waypoints = []
        for i in range(num_steps + 1):
            t = i / num_steps  # 插值参数，从0到1
            waypoint = (1 - t) * start_pos + t * end_pos
            waypoints.append(waypoint)
        return waypoints

        # 生成一系列waypoint
if __name__ == '__main__':
    env = BasicEnv()
    model_policy = "PPO"
    load_model_name = "npr5"
    model = PPO.load(f"./weights/{model_policy}/{load_model_name}", env=env, verbose=1)
    task = Task(env, model)
    task.move()
