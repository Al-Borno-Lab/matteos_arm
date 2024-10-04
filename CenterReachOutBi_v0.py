import collections
import gym
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from gym import error, logger, spaces
from gym.spaces import Space

class ReachEnvV0(BaseV0):
    DEFAULT_OBS_KEYS = ['hand_qpos', 'hand_qvel', 'obj_pos', 'reach_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "bonus": 4.0,
        "penalty": 50,
    }
    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)
    def _set_action_space(self):
        bounds = self.sim.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space
    def set_perturbation_force(self, force):
        self.perturbation_force = force
    def _setup(self,
               obj_xyz_range=None,
               far_th=.35, # set how far the hand can get from the target before it loses reward
               obs_keys: list = DEFAULT_OBS_KEYS,
               drop_th=0.50,
               qpos_noise_range=None,
               noise_std=0.02,
               weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs,
               ):
        self.far_th = far_th
        self.noise_std = noise_std
        self.palm_sid = self.sim.model.site_name2id("handsite") # site of hand
        self.object_sid = self.sim.model.site_name2id("object_o")
        self.object_bid = self.sim.model.body_name2id("Object") # site of object
        self.obj_xyz_range = obj_xyz_range
        self.drop_th = drop_th  
        self.qpos_noise_range = qpos_noise_range
        self.running_reach_cost = 0
        self.perturbation_flag = False
        self.perturbation_time = 0.0
        self.perturbation_force = 0
        
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs,
                       )
        keyFrame_id = 0
        #if self.obj_xyz_range is None else 1
        
        #print(self.sim.model.key_qpos)
        #print(keyFrame_id)
        #self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy() #Initial position for the hand
        #print(self.init_qpos[:])
        self.init_qpos[:] = [-0.785398163397, 1.57079632679]
        self._set_action_space()

    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['hand_qpos'] = self.sim.data.qpos[:].copy()
        self.obs_dict['hand_qvel'] = self.sim.data.qvel[:].copy() * self.dt
        if self.sim.model.na > 0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        # reach error
        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['palm_pos'] = sim.data.site_xpos[self.palm_sid]
        self.obs_dict['obj_pos'][2]=0
        self.obs_dict['palm_pos'][2]=0
        self.obs_dict['reach_err'] = np.array(self.obs_dict['palm_pos']) - np.array(self.obs_dict['obj_pos'])
        #print(self.obs_dict['reach_err'])
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys) #obs vector set up
        return obs


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[:].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[:].copy() * self.dt
        if sim.model.na > 0:
            obs_dict['act'] = sim.data.act[:].copy()

        # reach error
        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['palm_pos'] = sim.data.site_xpos[self.palm_sid]
        obs_dict['obj_pos'][2]=0
        obs_dict['palm_pos'][2]=0
        obs_dict['reach_err'] = np.array(obs_dict['palm_pos']) - np.array(obs_dict['obj_pos'])
        #print(obs_dict['reach_err'])
        return obs_dict # provide the obs_dict to the user
    

    def get_reward_dict(self, obs_dict): # set up reward dictionary
        reach_dist = abs(np.linalg.norm(obs_dict['reach_err'], axis=-1))
        act_mag = abs(np.linalg.norm(self.obs_dict['act'], axis=-1) / self.sim.model.na if self.sim.model.na != 0 else 0)
        far_th = abs(self.far_th)
        end_vel = abs(np.sqrt((np.array(self.obs_dict['hand_qvel'])[0][0][0])**2 + (np.array(self.obs_dict['hand_qvel'])[0][0][1])**2))
        near_th = 0.05
        drop = reach_dist > self.drop_th
        palm = np.array(self.obs_dict['palm_pos'])
        x_correct = ((palm[0][0][0] >= (self.obj_xyz_range[0][0])) and (palm[0][0][0] <= (self.obj_xyz_range[0][0]+near_th)))
        y_correct = ((palm[0][0][1] >= (self.obj_xyz_range[0][1] - near_th/2)) and (palm[0][0][1] <= (self.obj_xyz_range[0][1] + near_th/2)))
        # set up the reward dictionary
        self.running_reach_cost +=  -1. * reach_dist
        rwd_dict = collections.OrderedDict((
            ('reach', -1. * reach_dist),
            ('bonus', 1. * (reach_dist < near_th) and end_vel < 0.05),
            #('refund', reward_refund*0.5),
            ('act_reg', -1. * act_mag),
            ('penalty', -1. * (reach_dist > far_th) - end_vel*(x_correct and y_correct)*14),
            ('sparse', -1. * reach_dist),
            ('solved', reach_dist < near_th),
            ('done', x_correct and y_correct)
            #('done', x_correct and y_correct and end_vel < 0.05)
        ))
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict # return the reward dictionary to the user  ``
    # generate a valid target
    def generate_target_pose(self):
        random_index = np.random.randint(0, len(self.obj_xyz_range))
        self.sim.model.body_pos[self.object_bid] = self.obj_xyz_range[random_index]
        self.sim.forward() # step the simulation forward

    def reset(self, reset_qpos=None, reset_qvel=None):
        self.running_reach_cost = 0
        # randomize init arms pose
        if self.qpos_noise_range is not None:
            reset_qpos_local = self.init_qpos + self.qpos_noise_range*(self.sim.model.jnt_range[:,1]-self.sim.model.jnt_range[:,0])
            reset_qpos_local[-6:] = self.init_qpos[-6:]
        else:
            reset_qpos_local = reset_qpos
        self.generate_target_pose()
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset()
        self.perturbation_flag = True
        return obs
    
    def step(self, a, **kwargs):
        self.noisy = np.random.normal(0, self.noise_std, size=a.shape)
        a += self.noisy
                
        muscle_a = a.copy() # copy the muscle
        
        if((self.time>0) and self.perturbation_flag == True):
            #self.perturbation_flag = False
            xfrc_applied = self.sim.data.xfrc_applied.copy()
            xfrc_applied[5,1] = self.perturbation_force #hand
            self.sim.data.xfrc_applied = xfrc_applied
        # Explicitely project normalized space (-1,1) to actuator space (0,1) if muscles
        if self.sim.model.na and self.normalize_act:
            # find muscle actuators
            muscle_act_ind = self.sim.model.actuator_dyntype==3
            muscle_a[muscle_act_ind] = 1.0/(1.0+np.exp(-5.0*(muscle_a[muscle_act_ind]-0.5)))
            # TODO: actuator space may not always be (0,1) for muscle or (-1, 1) for others
            isNormalized = False # refuse internal reprojection as we explicitely did it here
        else:
            isNormalized = self.normalize_act # accept requested reprojection

        # implement abnormalities
        if self.muscle_condition == 'fatigue':
            for mus_idx in range(self.sim.model.actuator_gainprm.shape[0]):

                if self.sim.data.actuator_moment.shape[1]==1:
                    self.f_load[mus_idx].append(self.sim.data.actuator_moment[mus_idx].copy())
                else:
                    self.f_load[mus_idx].append(self.sim.data.actuator_moment[mus_idx,1].copy())

                if self.MVC_rest[mus_idx] != 0:
                    f_int = np.sum(self.f_load[mus_idx]-np.max(self.f_load[mus_idx],0),0)/self.MVC_rest[mus_idx]
                    f_cem = self.MVC_rest[mus_idx]*np.exp(self.k_fatigue*f_int)
                else:
                    f_cem = 0
                self.sim.model.actuator_gainprm[mus_idx,2] = f_cem
                self.sim_obsd.model.actuator_gainprm[mus_idx,2] = f_cem
        elif self.muscle_condition == 'reafferentation':
            # redirect EIP --> EPL
            muscle_a[self.EPLpos] = muscle_a[self.EIPpos].copy()
            # Set EIP to 0
            muscle_a[self.EIPpos] = 0

        # step forward
        self.last_ctrl = self.robot.step(ctrl_desired=muscle_a,
                                        ctrl_normalized=isNormalized,
                                        step_duration=0.01,
                                        realTimeSim=self.mujoco_render_frames,
                                        render_cbk=self.mj_render if self.mujoco_render_frames else None)

        # observation
        obs = self.get_obs(**kwargs)

        # rewards
        self.expand_dims(self.obs_dict) # required for vectorized rewards calculations
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.squeeze_dims(self.rwd_dict)
        self.squeeze_dims(self.obs_dict)

        # finalize step
        env_info = self.get_env_infos()

        # returns obs(t+1), rwd(t+1), done(t+1), info(t+1)
        return obs, env_info['rwd_'+self.rwd_mode], self.rwd_dict["done"], env_info
        #return obs, env_info['rwd_'+self.rwd_mode], bool(env_info['done']), env_info