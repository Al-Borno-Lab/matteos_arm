import collections, gym, numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
class ReachEnvV0(BaseV0):
    DEFAULT_OBS_KEYS = ['hand_qpos', 'hand_qvel', 'obj_pos', 'reach_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {"reach": 1.0, "bonus": 4.0, "penalty": 50,}
    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)
    def _setup(self, obj_xyz_range=None, far_th=.35, obs_keys: list = DEFAULT_OBS_KEYS, drop_th=0.50, qpos_noise_range=None, weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS, **kwargs,):
        self.far_th = far_th
        self.palm_sid = self.sim.model.site_name2id("handsite")
        self.object_sid = self.sim.model.site_name2id("object_o")
        self.object_bid = self.sim.model.body_name2id("Object")
        self.obj_xyz_range = obj_xyz_range
        self.drop_th = drop_th
        self.qpos_noise_range = qpos_noise_range
        self.running_reach_cost = 0
        super()._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, **kwargs,)
        keyFrame_id = 0 
        self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy()
    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['hand_qpos'] = self.sim.data.qpos[:].copy()
        self.obs_dict['hand_qvel'] = self.sim.data.qvel[:].copy() * self.dt
        if self.sim.model.na > 0: self.obs_dict['act'] = self.sim.data.act[:].copy()
        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['palm_pos'] = sim.data.site_xpos[self.palm_sid]
        self.obs_dict['obj_pos'][2]=0
        self.obs_dict['palm_pos'][2]=0
        self.obs_dict['reach_err'] = np.array(self.obs_dict['palm_pos']) - np.array(self.obs_dict['obj_pos'])
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs
    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[:].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[:].copy() * self.dt
        if sim.model.na > 0: obs_dict['act'] = sim.data.act[:].copy()
        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['palm_pos'] = sim.data.site_xpos[self.palm_sid]
        obs_dict['obj_pos'][2]=0
        obs_dict['palm_pos'][2]=0
        obs_dict['reach_err'] = np.array(obs_dict['palm_pos']) - np.array(obs_dict['obj_pos'])
        return obs_dict
    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1) / self.sim.model.na if self.sim.model.na != 0 else 0
        far_th = self.far_th
        end_vel = np.sqrt((np.array(self.obs_dict['hand_qvel'])[0][0][0])**2 + (np.array(self.obs_dict['hand_qvel'])[0][0][1])**2)
        near_th = 0.1
        reward_refund = 0
        drop = reach_dist > self.drop_th
        self.running_reach_cost +=  -1. * reach_dist
        if reach_dist < near_th: reward_refund = self.running_reach_cost
        rwd_dict = collections.OrderedDict((('reach', -1. * reach_dist), ('bonus', 1. * (reach_dist < 2 * near_th) + 1. * (reach_dist < near_th)), ('refund', reward_refund), ('act_reg', -1. * act_mag), ('penalty', -1. * (reach_dist > far_th)), ('sparse', -1. * reach_dist), ('solved', reach_dist < near_th), ('done', reach_dist < near_th)))
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict
    def generate_target_pose(self):
        random_index = np.random.randint(0, len(self.obj_xyz_range))
        self.sim.model.body_pos[self.object_bid] = self.obj_xyz_range[random_index]
        self.sim.forward()
    def reset(self, reset_qpos=None, reset_qvel=None):
        self.running_reach_cost = 0
        if self.qpos_noise_range is not None:
            reset_qpos_local = self.init_qpos + self.qpos_noise_range*(self.sim.model.jnt_range[:,1]-self.sim.model.jnt_range[:,0])
            reset_qpos_local[-6:] = self.init_qpos[-6:]
        else: reset_qpos_local = reset_qpos
        self.generate_target_pose()
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset()
        return obs
    def step(self, a, **kwargs):
        muscle_a = a.copy()
        if self.sim.model.na and self.normalize_act:
            muscle_act_ind = self.sim.model.actuator_dyntype==3
            muscle_a[muscle_act_ind] = 1.0/(1.0+np.exp(-5.0*(muscle_a[muscle_act_ind]-0.5)))
            isNormalized = False
        else: isNormalized = self.normalize_act
        if self.muscle_condition == 'fatigue':
            for mus_idx in range(self.sim.model.actuator_gainprm.shape[0]):
                if self.sim.data.actuator_moment.shape[1]==1:
                    self.f_load[mus_idx].append(self.sim.data.actuator_moment[mus_idx].copy())
                else: self.f_load[mus_idx].append(self.sim.data.actuator_moment[mus_idx,1].copy())
                if self.MVC_rest[mus_idx] != 0:
                    f_int = np.sum(self.f_load[mus_idx]-np.max(self.f_load[mus_idx],0),0)/self.MVC_rest[mus_idx]
                    f_cem = self.MVC_rest[mus_idx]*np.exp(self.k_fatigue*f_int)
                else: f_cem = 0
                self.sim.model.actuator_gainprm[mus_idx,2] = f_cem
                self.sim_obsd.model.actuator_gainprm[mus_idx,2] = f_cem
        elif self.muscle_condition == 'reafferentation':
            muscle_a[self.EPLpos] = muscle_a[self.EIPpos].copy()
            muscle_a[self.EIPpos] = 0
        self.last_ctrl = self.robot.step(ctrl_desired=muscle_a, ctrl_normalized=isNormalized, step_duration=0.01, realTimeSim=self.mujoco_render_frames, render_cbk=self.mj_render if self.mujoco_render_frames else None)
        obs = self.get_obs(**kwargs)
        self.expand_dims(self.obs_dict)
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.squeeze_dims(self.rwd_dict)
        self.squeeze_dims(self.obs_dict)
        env_info = self.get_env_infos()
        return obs, env_info['rwd_'+self.rwd_mode], bool(env_info['done']), env_info