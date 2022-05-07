import numpy as np

from gym_compete.new_envs import mujoco_env

class MultiAgentScene(mujoco_env.MujocoEnv):

    def __init__(self, xml_path, n_agents):
        self.n_agents = n_agents
        self._mujoco_init = False
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        self._mujoco_init = True

    def simulate(self, actions):
        a = np.concatenate(actions, axis=0)
        self.do_simulation(a, self.frame_skip)

    def step(self, actions):
        '''
        Just to satisfy mujoco_init, should not be used
        '''
        assert not self._mujoco_init, '_step should not be called on Scene'
        return self._get_obs(), 0, False, None

    def _get_obs(self):
        '''
        Just to satisfy mujoco_init, should not be used
        '''
        assert not self._mujoco_init, '_get_obs should not be called on Scene'
        obs = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1) 
       
        qpos[25:26] = self.init_qpos[25:26] + self.np_random.uniform(size=1, low=-.3, high=.3) 
        # print(self.init_qpos[24:27])
        # qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.3, high=.3) 
        # qpos[27] = 1
        
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        # print(self.model.nv, 'model nv')
        # init_state = np.concatenate((self.init_qpos,self.init_qvel),axis=-1)
        # np.save('/Users/xuanchen/Desktop/backdoor/mdp_perturb/init_state.npy',init_state)
        return None

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 0.55
        # self.viewer.cam.distance = self.model.stat.extent * 0.65
        # self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -10
        # self.viewer.cam.distance = self.model.stat.extent * 0.4
        # self.viewer.cam.lookat[2] += 1.0
        # self.viewer.cam.elevation = -25
        # self.viewer.cam.azimuth = 0 if np.random.random() > 0.5 else 180
        self.viewer.cam.azimuth = 90
        # self.viewer.vopt.flags[8] = True
        # self.viewer.vopt.flags[9] = True
        rand = self.np_random.random()
        if rand < 0.33:
            self.viewer.cam.azimuth = 0
        elif 0.33 <= rand < 0.66:
            self.viewer.cam.azimuth = 90
        else:
            self.viewer.cam.azimuth = 180
