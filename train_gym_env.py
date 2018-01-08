import sys

from absl import flags
from baselines import deepq
from pysc2.env import sc2_env
import os
import datetime

from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat

import random
import sc2ai.deepq_learner_marine_attack
import sc2ai.wrapper.gym_env


step_mul = 8

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "CollectMineralShards",
                    "Name of a map to use to play.")
start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
flags.DEFINE_string("log", "tensorboard", "logging type(stdout, tensorboard)")
flags.DEFINE_integer("timesteps", 2000000, "Steps to train")
flags.DEFINE_float("exploration_fraction", 0.3, "Exploration Fraction")
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

        # TODO: define this in one place
        ACTION_DO_NOTHING = 'donothing'
        ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
        ACTION_BUILD_BARRACKS = 'buildbarracks'
        ACTION_BUILD_MARINE = 'buildmarine'
        ACTION_ATTACK = 'attack'

        action_lib = [
            ACTION_DO_NOTHING,
            ACTION_BUILD_SUPPLY_DEPOT,
            ACTION_BUILD_BARRACKS,
            ACTION_BUILD_MARINE,
        ]

        for mm_x in range(1, 16):  # range should match minimap resolution
            for mm_y in range(1, 16):
                action_lib.append(ACTION_ATTACK + "_" + str(mm_x * 4) + "_" + str(mm_y * 4))

        gym_env = sc2ai.wrapper.gym_env.GymEnv(env=env,
                                               action_lookup=action_lib)

        model = deepq.models.mlp([512, 256, 64])

        act = deepq.learn(
            gym_env,
            q_func=model,
            lr=FLAGS.lr,
            max_timesteps=FLAGS.timesteps,
            buffer_size=10000,
            exploration_fraction=FLAGS.exploration_fraction,
            train_freq=4,
            print_freq=10,
            prioritized_replay=True,
            param_noise=True)
        act.save("gym_env_marine.pkl")


if __name__ == '__main__':
    main()
