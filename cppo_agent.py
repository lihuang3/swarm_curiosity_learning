import time

import numpy as np
import tensorflow as tf
from baselines.common import explained_variance
from baselines.common.mpi_moments import mpi_moments
from baselines.common.running_mean_std import RunningMeanStd
from mpi4py import MPI

from mpi_utils import MpiAdamOptimizer
from rollouts import Rollout
from utils import bcast_tf_vars_from_root, get_mean_and_std
from vec_env import ShmemVecEnv as VecEnv

getsess = tf.get_default_session


class PpoOptimizer(object):
    envs = None

    def __init__(self, *, hps, scope, ob_space, ac_space, stochpol,
                 ent_coef, gamma, gamma_ext, lam, nepochs, lr, cliprange,
                 nminibatches,
                 normrew, normadv, use_news, ext_coeff, int_coeff,
                 nsteps_per_seg, nsegs_per_env, dynamics):
        self.dynamics = dynamics
        with tf.variable_scope(scope):
            self.hps = hps
            self.use_recorder = True
            self.n_updates = 0
            self.scope = scope
            self.ob_space = ob_space
            self.ac_space = ac_space
            self.stochpol = stochpol
            self.nepochs = nepochs
            self.lr = lr
            self.cliprange = cliprange
            self.nsteps_per_seg = nsteps_per_seg
            self.nsegs_per_env = nsegs_per_env
            self.nminibatches = nminibatches
            self.gamma = gamma
            self.gamma_ext = gamma_ext
            self.lam = lam
            self.normrew = normrew
            self.normadv = normadv
            self.use_news = use_news
            self.ext_coeff = ext_coeff
            self.int_coeff = int_coeff

            self.ph_adv = tf.placeholder(tf.float32, [None, None])        
     
            self.ph_ret_int = tf.placeholder(tf.float32, [None, None])            
            self.ph_ret_ext = tf.placeholder(tf.float32, [None, None])            
            self.ph_ret = tf.placeholder(tf.float32, [None, None])

            self.ph_rews = tf.placeholder(tf.float32, [None, None])
            self.ph_oldnlp = tf.placeholder(tf.float32, [None, None])
            self.ph_oldvpred = tf.placeholder(tf.float32, [None, None])
            self.ph_lr = tf.placeholder(tf.float32, [])
            self.ph_cliprange = tf.placeholder(tf.float32, [])
            neglogpac = self.stochpol.pd.neglogp(self.stochpol.ph_ac)
            entropy = tf.reduce_mean(self.stochpol.pd.entropy())
            
            vpred = self.stochpol.vpred
            
            if hps['num_vf']==2: 
                # Separate vf_loss for intrinsic and extrinsic rewards
                vf_loss_int = 0.5 * tf.reduce_mean(tf.square(self.stochpol.vpred_int - self.ph_ret_int))
                vf_loss_ext = 0.5 * tf.reduce_mean(tf.square(self.stochpol.vpred_ext - self.ph_ret_ext))
                vf_loss = vf_loss_int + vf_loss_ext
            else:
                vf_loss = 0.5 * tf.reduce_mean((vpred - self.ph_ret) ** 2)

            ratio = tf.exp(self.ph_oldnlp - neglogpac)  # p_new / p_old
            negadv = - self.ph_adv
            pg_losses1 = negadv * ratio
            pg_losses2 = negadv * tf.clip_by_value(ratio, 1.0 - self.ph_cliprange, 1.0 + self.ph_cliprange)
            pg_loss_surr = tf.maximum(pg_losses1, pg_losses2)
            pg_loss = tf.reduce_mean(pg_loss_surr)
            ent_loss = (- ent_coef) * entropy
            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.ph_oldnlp))
            clipfrac = tf.reduce_mean(tf.to_float(tf.abs(pg_losses2 - pg_loss_surr) > 1e-6))

            self.total_loss = pg_loss + ent_loss + vf_loss
            self.to_report = {'tot': self.total_loss, 'pg': pg_loss, 'vf': vf_loss, 'ent': entropy,
                              'approxkl': approxkl, 'clipfrac': clipfrac}

    def start_interaction(self, env_fns, dynamics, nlump=2):
        self.loss_names, self._losses = zip(*list(self.to_report.items()))

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if MPI.COMM_WORLD.Get_size() > 1:
            trainer = MpiAdamOptimizer(learning_rate=self.ph_lr, comm=MPI.COMM_WORLD)
        else:
            trainer = tf.train.AdamOptimizer(learning_rate=self.ph_lr)
        gradsandvars = trainer.compute_gradients(self.total_loss, params)
        self._train = trainer.apply_gradients(gradsandvars)

        if MPI.COMM_WORLD.Get_rank() == 0:
            getsess().run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        bcast_tf_vars_from_root(getsess(), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        self.all_visited_rooms = []
        self.all_scores = []
        self.nenvs = nenvs = len(env_fns)
        self.nlump = nlump
        self.lump_stride = nenvs // self.nlump
        self.envs = [
            VecEnv(env_fns[l * self.lump_stride: (l + 1) * self.lump_stride], spaces=[self.ob_space, self.ac_space]) for
            l in range(self.nlump)]

        self.rollout = Rollout(hps=self.hps, ob_space=self.ob_space, ac_space=self.ac_space, nenvs=nenvs,
                               nsteps_per_seg=self.nsteps_per_seg,
                               nsegs_per_env=self.nsegs_per_env, nlumps=self.nlump,
                               envs=self.envs,
                               policy=self.stochpol,
                               int_rew_coeff=self.int_coeff,
                               ext_rew_coeff=self.ext_coeff,
                               record_rollouts=self.use_recorder,
                               dynamics=dynamics)

        self.buf_advs = np.zeros((nenvs, self.rollout.nsteps), np.float32)
        self.buf_advs_int = np.zeros((nenvs, self.rollout.nsteps), np.float32)
        self.buf_advs_ext = np.zeros((nenvs, self.rollout.nsteps), np.float32)

        self.buf_rets = np.zeros((nenvs, self.rollout.nsteps), np.float32)
        self.buf_rets_int = np.zeros((nenvs, self.rollout.nsteps), np.float32)
        self.buf_rets_ext = np.zeros((nenvs, self.rollout.nsteps), np.float32)


        if self.normrew:
            self.rff = RewardForwardFilter(self.gamma)
            self.rff_rms = RunningMeanStd()

        self.step_count = 0
        self.t_last_update = time.time()
        self.t_start = time.time()

    def stop_interaction(self):
        for env in self.envs:
            env.close()


    def update(self):
        # Rewards normalization
        # if self.normrew:
        #     rffs = np.array([self.rff.update(rew) for rew in self.rollout.buf_rews.T])
        #     rffs_mean, rffs_std, rffs_count = mpi_moments(rffs.ravel())
        #     self.rff_rms.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
        #     rews = self.rollout.buf_rews / np.sqrt(self.rff_rms.var)
        
        # Intrinsic Rewards Normalization
        if self.normrew:
            rffs_int = np.array([self.rff.update(rew) for rew in self.rollout.buf_int_rews.T])
            self.rff_rms.update(rffs_int.ravel())        
            int_rews = self.rollout.buf_int_rews / np.sqrt(self.rff_rms.var)
        else:
            int_rews = np.copy(self.rollout.buf_int_rews)
        
        mean_int_rew = np.mean(int_rews)
        max_int_rew = np.max(int_rews)
        
        # Do not normalize extrinsic rewards 
        ext_rews = self.rollout.buf_ext_rews

        nsteps = self.rollout.nsteps

        # If separate value fcn are used
        if self.hps['num_vf']==2:
            #Calculate intrinsic returns and advantages.
            lastgaelam = 0
            for t in range(nsteps - 1, -1, -1):  # nsteps-2 ... 0
                if self.use_news:
                    nextnew = self.rollout.buf_news[:, t + 1] if t + 1 < nsteps else self.rollout.buf_new_last
                else:
                    nextnew = 0 # No dones for intrinsic rewards with self.use_news=False
                nextvals = self.rollout.buf_vpreds_int[:, t + 1] if t + 1 < nsteps else self.rollout.buf_vpred_int_last
                nextnotnew = 1 - nextnew
                delta = int_rews[:, t] + self.gamma * nextvals * nextnotnew - self.rollout.buf_vpreds_int[:, t]
                self.buf_advs_int[:, t] = lastgaelam = delta + self.gamma * self.lam * nextnotnew * lastgaelam
            self.buf_rets_int[:] = self.buf_advs_int + self.rollout.buf_vpreds_int

            #Calculate extrinsic returns and advantages.
            lastgaelam = 0

            for t in range(nsteps - 1, -1, -1):  # nsteps-2 ... 0
                nextnew = self.rollout.buf_news[:, t + 1] if t + 1 < nsteps else self.rollout.buf_new_last
                nextvals = self.rollout.buf_vpreds_ext[:, t + 1] if t + 1 < nsteps else self.rollout.buf_vpred_ext_last
                nextnotnew = 1 - nextnew
                delta = ext_rews[:, t] + self.gamma_ext * nextvals * nextnotnew - self.rollout.buf_vpreds_ext[:, t]
                self.buf_advs_ext[:, t] = lastgaelam = delta + self.gamma_ext * self.lam * nextnotnew * lastgaelam
            self.buf_rets_ext[:] = self.buf_advs_ext + self.rollout.buf_vpreds_ext
            
            #Combine the extrinsic and intrinsic advantages.
            self.buf_advs = self.int_coeff*self.buf_advs_int + self.ext_coeff*self.buf_advs_ext
        else:
            #Calculate mixed intrinsic and extrinsic returns and advantages.
            rews = self.rollout.buf_rews = self.rollout.reward_fun(int_rew=int_rews, ext_rew=ext_rews)            
            lastgaelam = 0
            for t in range(nsteps - 1, -1, -1):  # nsteps-2 ... 0
                nextnew = self.rollout.buf_news[:, t + 1] if t + 1 < nsteps else self.rollout.buf_new_last
                nextvals = self.rollout.buf_vpreds[:, t + 1] if t + 1 < nsteps else self.rollout.buf_vpred_last
                nextnotnew = 1 - nextnew
                delta = rews[:, t] + self.gamma * nextvals * nextnotnew - self.rollout.buf_vpreds[:, t]
                self.buf_advs[:, t] = lastgaelam = delta + self.gamma * self.lam * nextnotnew * lastgaelam
            self.buf_rets[:] = self.buf_advs + self.rollout.buf_vpreds
        
        info = dict(
            # advmean=self.buf_advs.mean(),
            # advstd=self.buf_advs.std(),  
            recent_best_ext_ret=self.rollout.current_max,
            recent_best_eplen = self.rollout.current_minlen,
            recent_worst_eplen = self.rollout.current_maxlen   
        )

        if self.hps['num_vf'] ==2:
            info['retmean_int']=self.buf_rets_int.mean()
            info['retmean_ext']=self.buf_rets_ext.mean()
            info['retstd_int']=self.buf_rets_int.std()
            info['retstd_ext']=self.buf_rets_ext.std()
            info['vpredmean_int']=self.rollout.buf_vpreds_int.mean()
            info['vpredmean_ext']=self.rollout.buf_vpreds_ext.mean()
            info['vpredstd_int']=self.rollout.buf_vpreds_int.std()
            info['vpredstd_ext']=self.rollout.buf_vpreds_ext.std()
            info['ev_int']=explained_variance(self.rollout.buf_vpreds_int.ravel(), self.buf_rets_int.ravel())            
            info['ev_ext']=explained_variance(self.rollout.buf_vpreds_ext.ravel(), self.buf_rets_ext.ravel())            
            info['rew_int_mean']=mean_int_rew
            info['recent_best_int_rew']=max_int_rew
        else:
            # info['retmean']=self.buf_rets.mean()
            # info['retstd']=self.buf_rets.std()
            # info['vpredmean']=self.rollout.buf_vpreds.mean()
            # info['vpredstd']=self.rollout.buf_vpreds.std()
            info['rew_mean']=np.mean(self.rollout.buf_rews)
            info['eplen_std']=np.std(self.rollout.statlists['eplen'])            
            info['eprew_std']=np.std(self.rollout.statlists['eprew'])
            # info['ev']=explained_variance(self.rollout.buf_vpreds.ravel(), self.buf_rets.ravel())            

        if self.rollout.best_ext_ret is not None:
            info['best_ext_ret'] = self.rollout.best_ext_ret
            info['best_eplen'] = self.rollout.best_eplen

        # normalize advantages
        if self.normadv:
            m, s = get_mean_and_std(self.buf_advs)
            self.buf_advs = (self.buf_advs - m) / (s + 1e-7)
        envsperbatch = (self.nenvs * self.nsegs_per_env) // self.nminibatches
        envsperbatch = max(1, envsperbatch)
        envinds = np.arange(self.nenvs * self.nsegs_per_env)

        def resh(x):
            if self.nsegs_per_env == 1:
                return x
            sh = x.shape
            return x.reshape((sh[0] * self.nsegs_per_env, self.nsteps_per_seg) + sh[2:])
        
        #Create feed_dict for optimization.
        ph_buf = [
                (self.stochpol.ph_ac, resh(self.rollout.buf_acs)),
                (self.ph_oldnlp, resh(self.rollout.buf_nlps)),
                (self.stochpol.ph_ob, resh(self.rollout.buf_obs)),
                (self.ph_adv, resh(self.buf_advs)),
                ]

        if self.hps['num_vf']==2:
            ph_buf.extend([                
                (self.ph_ret_int, resh(self.buf_rets_int)),
                (self.ph_ret_ext, resh(self.buf_rets_ext)),
            ])       
        else:
            ph_buf.extend([
                (self.ph_rews, resh(self.rollout.buf_rews)),
                (self.ph_oldvpred, resh(self.rollout.buf_vpreds)),
                (self.ph_ret, resh(self.buf_rets)),
            ])

        ph_buf.extend([
            (self.dynamics.last_ob,
             self.rollout.buf_obs_last.reshape([self.nenvs * self.nsegs_per_env, 1, *self.ob_space.shape]))
        ])

        #Optimizes on current data for several epochs.
        mblossvals = []

        for _ in range(self.nepochs):
            np.random.shuffle(envinds)
            for start in range(0, self.nenvs * self.nsegs_per_env, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                fd = {ph: buf[mbenvinds] for (ph, buf) in ph_buf}                
                fd.update({self.ph_lr: self.lr, self.ph_cliprange: self.cliprange})
                mblossvals.append(getsess().run(self._losses + (self._train,), fd)[:-1])

        mblossvals = [mblossvals[0]]
        # info.update(zip(['opt_' + ln for ln in self.loss_names], np.mean([mblossvals[0]], axis=0)))
        # info["rank"] = MPI.COMM_WORLD.Get_rank()
        self.n_updates += 1
        info["n_updates"] = self.n_updates
        info.update({dn: (np.mean(dvs) if len(dvs) > 0 else 0) for (dn, dvs) in self.rollout.statlists.items()})
        info.update(self.rollout.stats)
        if "states_visited" in info:
            info.pop("states_visited")
        tnow = time.time()
        # info["ups"] = 1. / (tnow - self.t_last_update)
        info["total_secs"] = tnow - self.t_start
        # info['tps'] = MPI.COMM_WORLD.Get_size() * self.rollout.nsteps * self.nenvs / (tnow - self.t_last_update)
        self.t_last_update = tnow

        return info

    def step(self):
        self.rollout.collect_rollout()
        update_info = self.update()
        return {'update': update_info}

    def get_var_values(self):
        return self.stochpol.get_var_values()

    def set_var_values(self, vv):
        self.stochpol.set_var_values(vv)


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems
