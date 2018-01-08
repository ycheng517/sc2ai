import numpy as np
import pprint
import random
from pysc2.env import base_env_wrapper
import gym
from pysc2.lib import actions
from pysc2.lib import features


ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'
ACTION_ATTACK_UP = 'attackup'
ACTION_ATTACK_DOWN = 'attackdown'
ACTION_ATTACK_LEFT = 'attackleft'
ACTION_ATTACK_RIGHT = 'attackright'
ACTION_MOVE = 'move'
ACTION_MOVE_UP = 'moveup'
ACTION_MOVE_DOWN = 'movedown'
ACTION_MOVE_LEFT = 'moveleft'
ACTION_MOVE_RIGHT = 'moveright'

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

_PLAYER_SELF = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_MARINE = 48
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id

_NO_OP = actions.FUNCTIONS.no_op.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_LOWER_SUPPLY_DEPOT = actions.FUNCTIONS.Morph_SupplyDepot_Lower_quick.id

_SELECTED = features.SCREEN_FEATURES.selected.index
_SELECTED_MINIMAP = features.MINIMAP_FEATURES.selected.index
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_MINIMAP = features.MINIMAP_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# player features
_ARMY_SUPPLY = 5

# score cumulative
_KILLED_VALUE_STRUCTURES = 6

def action_to_coord(action_name):
    _, x, y = action_name.split('_')
    return x, y


class GymActionSpace(gym.Space):
    def __init__(self, spec):
        self.n = len(spec)

    @property
    def shape(self):
        return (self.n,)


class GymObservationSpace(gym.Space):
    def __init__(self, spec):
        self.n = spec['screen'][1] * spec['screen'][2] + spec['minimap'][1] * spec['minimap'][2]

    @property
    def shape(self):
        return (self.n,)


class GymEnv(base_env_wrapper.BaseEnvWrapper):
    """An env wrapper to print the available actions."""

    def __init__(self, env, action_lookup):
        super(GymEnv, self).__init__(env)
        self._action_lookup = action_lookup
        self._action_spec = self.action_spec()
        self._observation_spec = self.observation_spec()
        print("ACTION SPEC:::::::::::::::")
        pprint.pprint(self._action_spec)

        self.observation_space = GymObservationSpace(self._observation_spec)
        self.action_space = GymActionSpace(self._action_lookup)
        self.prev_obs = None

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # TODO: move this out of here

        # get current obs
        obs = super(GymEnv, self).step(actions=[actions.FunctionCall(_NO_OP, [])])
        if obs[0].last():
            return self._gym_step_returns(obs)

        # track reward before action
        visibility = len(obs[0].observation['minimap'][features.MINIMAP_FEATURES.visibility_map.index].nonzero()[0])
        army_supply = obs[0].observation['player'][_ARMY_SUPPLY]
        killed_buildings = obs[0].observation['score_cumulative'][_KILLED_VALUE_STRUCTURES]

        pending_action = self._action_lookup[action]

        if pending_action == ACTION_DO_NOTHING:
            pass
        elif pending_action in (ACTION_BUILD_BARRACKS, ACTION_BUILD_SUPPLY_DEPOT):
            # move camera to main base
            player_y, player_x = (obs[0].observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            obs = super(GymEnv, self).step(actions=[actions.FunctionCall(_MOVE_CAMERA,
                                                                                 [[player_x.mean(), player_y.mean()]])])
            if obs[0].last():
                return self._gym_step_returns(obs)
            # select scv
            unit_type = obs[0].observation['screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]
                obs = super(GymEnv, self).step([actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])])
                if obs[0].last():
                    return self._gym_step_returns(obs)

                # build building
                target = [random.uniform(4, 80), random.uniform(4, 80)]
                if pending_action == ACTION_BUILD_BARRACKS and _BUILD_BARRACKS in obs[0].observation['available_actions']:
                    obs = super(GymEnv, self).step([actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])])
                elif pending_action == ACTION_BUILD_SUPPLY_DEPOT and _BUILD_SUPPLY_DEPOT in obs[0].observation['available_actions']:
                    obs = super(GymEnv, self).step([actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])])
                if obs[0].last():
                    return self._gym_step_returns(obs)

                # return to gathering minerals
                if _HARVEST_GATHER in obs[0].observation['available_actions']:
                    unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)
                        m_x = unit_x[i]
                        m_y = unit_y[i]
                        target = [int(m_x), int(m_y)]
                        obs = super(GymEnv, self).step([actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])])

        elif pending_action == ACTION_BUILD_MARINE:
            # move camera to main base
            player_y, player_x = (obs[0].observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            obs = super(GymEnv, self).step(actions=[actions.FunctionCall(_MOVE_CAMERA,
                                                                                 [[player_x.mean(), player_y.mean()]])])
            if obs[0].last():
                return self._gym_step_returns(obs)
            unit_type = obs[0].observation['screen'][_UNIT_TYPE]
            barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
            if barracks_y.any():
                i = random.randint(0, len(barracks_y) - 1)
                target = [barracks_x[i], barracks_y[i]]

                obs = super(GymEnv, self).step([actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])])
                if obs[0].last():
                    return self._gym_step_returns(obs)
                if _TRAIN_MARINE in obs[0].observation['available_actions']:
                    obs = super(GymEnv, self).step([actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])])

        elif ACTION_ATTACK in pending_action:
            if _SELECT_ARMY in obs[0].observation['available_actions']:
                obs = super(GymEnv, self).step([actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])])
                if obs[0].last():
                    return self._gym_step_returns(obs)

                do_it = True
                if len(obs[0].observation['single_select']) > 0 and obs[0].observation['single_select'][0][0] == _TERRAN_SCV:
                    do_it = False
                if len(obs[0].observation['multi_select']) > 0 and obs[0].observation['multi_select'][0][0] == _TERRAN_SCV:
                    do_it = False

                if do_it and _ATTACK_SCREEN in obs[0].observation["available_actions"]:
                    selected_y, selected_x = obs[0].observation['minimap'][_SELECTED_MINIMAP].nonzero()
                    target = [selected_x.mean(), selected_y.mean()]
                    obs = super(GymEnv, self).step(actions=[actions.FunctionCall(_MOVE_CAMERA, [target])])
                    if obs[0].last():
                        return self._gym_step_returns(obs)
                    if _ATTACK_SCREEN in obs[0].observation["available_actions"]:
                        units_y, units_x = obs[0].observation['screen'][_SELECTED].nonzero()
                        if units_y.any():
                            target = [units_x.mean(), units_y.mean()]
                            if pending_action == ACTION_ATTACK_UP:
                                target[1] += 10
                            elif pending_action == ACTION_ATTACK_DOWN:
                                target[1] -= 10
                            elif pending_action == ACTION_ATTACK_LEFT:
                                target[0] -= 10
                            elif pending_action == ACTION_ATTACK_RIGHT:
                                target[0] += 10
                            target[0] = max(min(target[0], 80), 0)
                            target[1] = max(min(target[1], 80), 0)
                            if obs[0].observation['screen'][_PLAYER_RELATIVE][int(target[1])][int(target[0])] != _PLAYER_SELF:
                                obs = super(GymEnv, self).step(actions=[actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])])

        elif ACTION_MOVE in pending_action:
            if _SELECT_ARMY in obs[0].observation['available_actions']:
                obs = super(GymEnv, self).step([actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])])
                if obs[0].last():
                    return self._gym_step_returns(obs)

                do_it = True
                if len(obs[0].observation['single_select']) > 0 and obs[0].observation['single_select'][0][0] == _TERRAN_SCV:
                    do_it = False
                if len(obs[0].observation['multi_select']) > 0 and obs[0].observation['multi_select'][0][0] == _TERRAN_SCV:
                    do_it = False

                if do_it and _MOVE_SCREEN in obs[0].observation["available_actions"]:
                    selected_y, selected_x = obs[0].observation['minimap'][_SELECTED_MINIMAP].nonzero()
                    target = [selected_x.mean(), selected_y.mean()]
                    obs = super(GymEnv, self).step(actions=[actions.FunctionCall(_MOVE_CAMERA, [target])])
                    if obs[0].last():
                        return self._gym_step_returns(obs)
                    if _MOVE_SCREEN in obs[0].observation["available_actions"]:
                        units_y, units_x = obs[0].observation['screen'][_SELECTED].nonzero()
                        if units_y.any():
                            target = [units_x.mean(), units_y.mean()]
                            if pending_action == ACTION_ATTACK_UP:
                                target[1] += 10
                            elif pending_action == ACTION_ATTACK_DOWN:
                                target[1] -= 10
                            elif pending_action == ACTION_ATTACK_LEFT:
                                target[0] -= 10
                            elif pending_action == ACTION_ATTACK_RIGHT:
                                target[0] += 10
                            target[0] = max(min(target[0], 80), 0)
                            target[1] = max(min(target[1], 80), 0)
                            obs = super(GymEnv, self).step(actions=[actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])])

        extra_reward = self._calculate_extra_reward(obs, visibility, army_supply, killed_buildings)
        return self._gym_step_returns(obs, extra_reward)

    def _gym_step_returns(self, obs, extra_reward=0):
        player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
        player_relative_minimap = obs[0].observation['minimap'][_PLAYER_RELATIVE_MINIMAP]
        obs_flattened = np.concatenate((np.array(player_relative).flatten(), np.array(player_relative_minimap).flatten()))
        rew = obs[0].reward + extra_reward
        if obs[0].last():
            print("reward: {}=======================".format(rew))
        return obs_flattened, rew, obs[0].last(), {}

    def _gym_reset_returns(self, obs):
        player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
        player_relative_minimap = obs[0].observation['minimap'][_PLAYER_RELATIVE_MINIMAP]
        obs_flattened = np.concatenate((np.array(player_relative).flatten(), np.array(player_relative_minimap).flatten()))
        return obs_flattened

    def reset(self):
        obs = super(GymEnv, self).reset()
        return self._gym_reset_returns(obs)

    def _calculate_extra_reward(self, obs, pre_visibility, pre_army_supply, pre_killed_buildings):
        visibility = len(obs[0].observation['minimap'][features.MINIMAP_FEATURES.visibility_map.index].nonzero()[0])
        army_supply = obs[0].observation['player'][_ARMY_SUPPLY]
        killed_buildings = obs[0].observation['score_cumulative'][_KILLED_VALUE_STRUCTURES]
        # print("visibility: pre: {}, post:{}", pre_visibility, visibility)
        # print("army supply: pre: {}, post:{}", pre_army_supply, army_supply)
        # print("killed buildins: pre: {}, post:{}", pre_killed_buildings, killed_buildings)
        rew = 0
        if visibility > pre_visibility:
            rew += 0.001
        if army_supply > pre_army_supply:
            rew += 0.002
        if killed_buildings > pre_killed_buildings:
            rew += 0.005
        return rew
