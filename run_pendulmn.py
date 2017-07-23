#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import tensorflow as tf

import tempfile
import zipfile
import dill

import numpy as np

from baselines.pposgd import mlp_policy

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=64, num_hid_layers=2)

import pposgd_simple

RECORD_MOVIE = False

if RECORD_MOVIE:
    import cv2

def replay(env_id, num_timesteps, seed):
    sess=U.make_session(num_cpu=1)
    sess.__enter__()

    set_global_seeds(seed)
    env = gym.make(env_id)
    env.seed(seed)

    # creating environment
    pi=pposgd_simple.create(env, policy_fn,clip_param=0.2, entcoeff=0.0)

    # restore policy
    saver = tf.train.Saver()
    saver.restore(sess, "model/model.ckpt")

    # replay
    env = gym.make(env_id)
    observation = env.reset()

    if RECORD_MOVIE:
        i=0

    for _ in range(num_timesteps):
      if RECORD_MOVIE:
       img=env.render(mode="rgb_array")
       img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
       cv2.imwrite("movie/%d.png"%i, img)
       i+=1
      else:
       env.render()
      action = pi.act(False,observation)[0]
      observation, reward, done, info = env.step(action)



def train(env_id, num_timesteps, seed):
    from baselines.pposgd import mlp_policy
    import pposgd_simple
    sess=U.make_session(num_cpu=1)
    sess.__enter__()
    logger.session().__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    env = bench.Monitor(env, "monitor.json")
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95,
             schedule='linear'
        )
    env.close()

    # save model
    saver = tf.train.Saver()
    saver.save(sess, "model/model.ckpt")


if __name__ == '__main__':
    if len(sys.argv)!=2:
        print("Usage: python run_pendulmn.py [train/replay]")
        exit()
    option=sys.argv[1]

    env_name = 'Pendulum-v0'

    if option=="train":
        train(env_name, num_timesteps=1e6, seed=0)
    elif option=="replay":
        replay(env_name, num_timesteps=100, seed=0)
    else:
        print("Unknown option: ",option)
