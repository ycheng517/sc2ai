import sys

from absl import flags
from baselines import deepq
from pysc2.env import sc2_env
import os
import datetime

from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat

import random
import sc2ai.deepq_learner_marine_attack


step_mul = 8

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "CollectMineralShards",
                    "Name of a map to use to play.")
start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
flags.DEFINE_string("log", "tensorboard", "logging type(stdout, tensorboard)")
flags.DEFINE_integer("timesteps", 2000000, "Steps to train")
flags.DEFINE_float("exploration_fraction", 0.5, "Exploration Fraction")
flags.DEFINE_boolean("prioritized", True, "prioritized_replay")
flags.DEFINE_boolean("dueling", True, "dueling")
flags.DEFINE_float("lr", 0.0005, "Learning rate")

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

max_mean_reward = 0
last_filename = ""

def main():
    FLAGS(sys.argv)

    print("timesteps : %s" % FLAGS.timesteps)
    print("exploration_fraction : %s" % FLAGS.exploration_fraction)
    print("prioritized : %s" % FLAGS.prioritized)
    print("dueling : %s" % FLAGS.dueling)
    print("lr : %s" % FLAGS.lr)

    if FLAGS.lr == 0:
        FLAGS.lr = random.uniform(0.00001, 0.001)

    print("random lr : %s" % FLAGS.lr)

    lr_round = round(FLAGS.lr, 8)

    logdir = "tensorboard/marine_attack/%s_%s_prio%s_duel%s_lr%s/%s" % (
        FLAGS.timesteps, FLAGS.exploration_fraction,
        FLAGS.prioritized, FLAGS.dueling, lr_round, start_time)

    if FLAGS.log == "tensorboard":
        Logger.DEFAULT \
            = Logger.CURRENT \
            = Logger(dir=None,
                     output_formats=[TensorBoardOutputFormat(logdir)])
    elif FLAGS.log == "stdout":
        Logger.DEFAULT \
            = Logger.CURRENT \
            = Logger(dir=None,
                     output_formats=[HumanOutputFormat(sys.stdout)])

    with sc2_env.SC2Env(
            map_name="Simple64",
            agent_race="T",
            step_mul=step_mul,
            visualize=True,
            screen_size_px=(84, 84),
            minimap_size_px=(64, 64)) as env:

        model = deepq.models.cnn_to_mlp(
            convs=[(16, 8, 4), (32, 4, 2)], hiddens=[256], dueling=True)

        act = sc2ai.deepq_learner_marine_attack.learn(
            env,
            q_func=model,
            lr=FLAGS.lr,
            max_timesteps=FLAGS.timesteps,
            buffer_size=10000,
            exploration_fraction=FLAGS.exploration_fraction,
            exploration_final_eps=0.01,
            train_freq=4,
            learning_starts=1000,
            target_network_update_freq=500,
            gamma=0.99,
            prioritized_replay=True,
            callback=deepq_callback)
        act.save("marine_attack.pkl")


def deepq_callback(locals, globals):
    global max_mean_reward, last_filename
    if 'done' in locals and locals['done']:
        if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 10
                and locals['mean_100ep_reward'] > max_mean_reward):
            print("mean_100ep_reward : %s max_mean_reward : %s" %
                  (locals['mean_100ep_reward'], max_mean_reward))

            if (not os.path.exists(
                    os.path.join(PROJ_DIR, 'models/marine_attack/'))):
                try:
                    os.mkdir(os.path.join(PROJ_DIR, 'models/'))
                except Exception as e:
                    print(str(e))
                try:
                    os.mkdir(os.path.join(PROJ_DIR, 'models/marine_attack/'))
                except Exception as e:
                    print(str(e))

            if last_filename != "":
                os.remove(last_filename)
                print("delete last model file : %s" % last_filename)

            max_mean_reward = locals['mean_100ep_reward']
            act = sc2ai.deepq_learner_marine_attack.ActWrapper(locals['act'], locals['act_params'])

            filename = os.path.join(PROJ_DIR,
                                    'models/marine_attack/reward_%s.pkl' %
                                    locals['mean_100ep_reward'])
            act.save(filename)
            print("save best mean_100ep_reward model to %s" % filename)
            last_filename = filename


if __name__ == '__main__':
    main()
