"""SAC from images with a demonstration buffer on the Isaac Lab cup scene (skrl), started by
cup_demos.py --sac / --sac-eval.

- Actor (the deployable policy): the same background-filtered static + wrist camera images as BAKU
  (sim2real/observation.py, resized to 84x84) + what the real robot also has (end-effector position,
  gripper width and command). Critic: the sim state (cup poses) + the actor's proprioception
  (asymmetric actor-critic; the critic is only used in training).
- Action (10 Hz, as BAKU): a target 5 cm x [dx dy dz] away from the current end-effector position, held
  for 3 control steps and tracked by the teleop law (cup_demos.TeleopTracking), gripper open/close with
  hysteresis. The gripper keeps the home orientation (the task never rotates it; on the robot that is a
  zero rotation action).
- Demonstrations: the scripted hand (cup_demos.py plan()) run through the same step function, only
  successful episodes, in a separate buffer. Every update batch is half demo, half online (RLPD) and the
  actor loss has a behavior-cloning term on the demo half (SACfD below); the actor is first fit to the
  demos (BC) so RL starts from a policy that often succeeds.
- Reward (sim state): reach the picked cup, hold it lifted, bring it over the other cup; when nested and
  released +20 plus what holding could still have earned (so releasing beats hovering); -5 and the episode
  ends when a cup tips/moves away or the hand leaves the workspace.
"""
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.agents.torch import Agent
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim2real.observation import BackgroundFilter  # noqa: E402
from sim2real.sac_policy import IMG, N_PROPRIO, STEP, Encoder  # noqa: E402  (the deployable actor's parts)

N_IMG = 2 * 3 * IMG * IMG   # static + wrist camera, uint8 values
N_PRIV = 14                 # sim state, critic only
N_VEC = N_PROPRIO + N_PRIV
DECISION = 3                # 30 Hz control steps per action (10 Hz)


# ------------------------------------------------------------------ environment
class CupEnv(Wrapper):
    def __init__(self, api, horizon=250):
        super().__init__(api)
        self.api, self.n, self.horizon = api, api.n, horizon
        self._obs_space = gymnasium.spaces.Box(-np.inf, np.inf, (N_IMG + N_VEC,))
        self._act_space = gymnasium.spaces.Box(-1.0, 1.0, (4,))
        self.bg = BackgroundFilter(api.data / 'scene.json', api.data / 'calibration.json', sim=True)
        self.pool = ThreadPoolExecutor(16)
        self.hb, self.hr = api.sizes[api.picked][2], api.sizes[api.target][2]
        self.steps = np.zeros(self.n, int)
        self.open = np.ones(self.n, bool)
        self.targets = np.tile(api.home_ee, (self.n, 1, 1))
        self.target0 = np.zeros((self.n, 3))
        self.returns = np.zeros(self.n)
        self.finished = []          # (success, return, steps) of finished episodes
        self.scripted = None        # per-environment scripted hand while collecting demonstrations
        self.last_print = time.time()

    @property
    def num_envs(self):
        return self.n

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._act_space

    def close(self):
        pass

    # -- episodes
    def _reset(self, ids):
        ids = list(ids)
        positions = [self.api.place_cups() for _ in ids]
        self.api.reset_envs(ids, positions)
        self.api.tracker.reset(ids, np.tile(self.api.home_ee, (len(ids), 1, 1)))
        self.targets[ids] = self.api.home_ee
        self.open[ids] = True
        self.steps[ids] = 0
        self.returns[ids] = 0
        for i, p in zip(ids, positions):
            self.target0[i, :2] = p[self.api.target]
            if self.scripted is not None:
                self.scripted[i] = {'segments': self.api.plan(p), 'index': 0, 'k': 0}

    def reset(self):
        self._reset(range(self.n))
        self.api.control(self.targets, self.open)
        obs, _ = self._observe()
        return obs, {}

    def step(self, actions):
        a = actions.detach().clamp(-1, 1).cpu().numpy()
        ee = self.api.ee_poses()
        self.targets = np.tile(self.api.home_ee, (self.n, 1, 1))  # home orientation: the gripper stays level
        self.targets[:, :3, 3] = ee[:, :3, 3] + a[:, :3] * STEP
        self.open = np.where(a[:, 3] > 0.3, True, np.where(a[:, 3] < -0.3, False, self.open))
        for _ in range(DECISION):
            self.api.control(self.targets, self.open)
        self.steps += 1
        obs, state = self._observe()
        reward, terminated, success = self._reward(state)
        truncated = self.steps >= self.horizon
        self.returns += reward
        done = terminated | truncated
        if done.any():
            ids = np.nonzero(done)[0]
            self.finished += [(bool(success[i]), float(self.returns[i]), int(self.steps[i])) for i in ids]
            self._reset(ids)
            self.api.control(self.targets, self.open)  # one step so the reset environments are rendered
            obs, _ = self._observe()
        if time.time() - self.last_print > 60 and self.finished:
            recent = self.finished[-100:]
            print('[%s] episodes %d; last %d: success %.0f%%, return %.1f, length %.0f steps' % (
                time.strftime('%H:%M:%S'), len(self.finished), len(recent), 100 * np.mean([r[0] for r in recent]),
                np.mean([r[1] for r in recent]), np.mean([r[2] for r in recent])), flush=True)
            self.last_print = time.time()
        t = lambda x, dtype=torch.float32: torch.tensor(x, dtype=dtype, device=self.device)[:, None]  # noqa: E731
        return obs, t(reward), t(terminated, torch.bool), t(truncated, torch.bool), {'success': success}

    # -- observation and reward
    def _observe(self):
        api = self.api
        ee = api.ee_poses()
        cams = {}
        for name, cam in (('cam_2', api.static_cam), ('cam_1', api.wrist_cam)):
            rgb = cam.data.output['rgb'][..., :3].contiguous().cpu().numpy()
            depth = torch.nan_to_num(cam.data.output['distance_to_image_plane'][..., 0] * 1000, posinf=0)
            cams[name] = (rgb, depth.clamp(0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16))

        def image(i):
            out = []
            for name in ('cam_2', 'cam_1'):
                rgb, depth = cams[name]
                filtered = self.bg.apply(name, rgb[i], depth[i], ee[i])
                out.append(cv2.resize(filtered, (IMG, IMG), interpolation=cv2.INTER_AREA).transpose(2, 0, 1))
            return np.concatenate(out)

        images = np.stack(list(self.pool.map(image, range(self.n))))  # (n, 6, IMG, IMG) uint8
        width = api.gripper_width()
        picked, target, up_picked, up_target = api.cup_states()
        e = ee[:, :3, 3]
        vec = np.concatenate([e, width[:, None], self.open[:, None].astype(float),
                              picked, target, up_picked[:, None], up_target[:, None], picked - e, target - picked], 1)
        if not np.isfinite(vec).all():
            bad = sorted(set(np.nonzero(~np.isfinite(vec))[1].tolist()))
            print('non-finite observation columns %s' % bad, flush=True)
        obs = torch.cat([torch.tensor(images.reshape(self.n, -1), device=self.device, dtype=torch.float32),
                         torch.tensor(vec, device=self.device, dtype=torch.float32)], 1)
        return obs, {'ee': e, 'picked': picked, 'target': target, 'up_picked': up_picked, 'up_target': up_target,
                     'images': images}

    def _reward(self, s):
        e, p, t = s['ee'], s['picked'], s['target']
        grasp = p + np.array([0, 0, self.hb - 0.025])
        reach = np.linalg.norm(e - grasp, axis=1)
        offset = np.linalg.norm(p[:, :2] - t[:, :2], axis=1)
        holding = (p[:, 2] > 0.02) & (reach < 0.05) & ~self.open
        nested = (offset < 0.02) & (s['up_picked'] > 0.9) & (p[:, 2] - t[:, 2] < 0.6 * self.hr)
        success = nested & self.open
        fail = ((s['up_target'] < 0.9) | (np.linalg.norm(t[:, :2] - self.target0[:, :2], axis=1) > 0.03)
                | ((s['up_picked'] < 0.7) & ~nested)
                | (e[:, 0] < 0.15) | (e[:, 0] > 0.8) | (np.abs(e[:, 1]) > 0.45) | (e[:, 2] < 0.01) | (e[:, 2] > 0.65))
        fail &= ~success
        # success also pays what holding could still have earned until the horizon, so releasing a nested cup
        # always beats hovering with it
        bonus = 20.0 + 0.6 * (self.horizon - self.steps)
        reward = (0.1 * (1 - np.tanh(10 * reach)) + 0.2 * holding + 0.3 * holding * (1 - np.tanh(10 * offset))
                  + bonus * success - 5.0 * fail)
        return reward, success | fail, success

    # -- the scripted hand as actions of this environment
    def scripted_actions(self):
        ee = self.api.ee_poses()
        actions = np.zeros((self.n, 4))
        for i, s in enumerate(self.scripted):
            target, opened = None, True
            for _ in range(DECISION):
                if s['index'] >= len(s['segments']):
                    break
                kind, *seg = s['segments'][s['index']]
                s['k'] += 1
                if kind == 'move':
                    target, opened = seg[0][min(s['k'], len(seg[0])) - 1], seg[1]
                    done = s['k'] >= len(seg[0])
                elif kind == 'wait':
                    target, opened = seg[0], seg[1]
                    done = np.linalg.norm(ee[i, :3, 3] - seg[0][:3, 3]) < 0.004 or s['k'] >= 60
                else:
                    target, opened = seg[0], seg[1]
                    done = s['k'] >= seg[2]
                if done:
                    s['index'], s['k'] = s['index'] + 1, 0
            if target is None:  # plan finished: stay, open
                target = s['segments'][-1][1] if s['segments'][-1][0] != 'move' else s['segments'][-1][1][-1]
            actions[i, :3] = np.clip((target[:3, 3] - ee[i, :3, 3]) / STEP, -1, 1)
            actions[i, 3] = 1.0 if opened else -1.0
        return torch.tensor(actions, dtype=torch.float32, device=self.device)


# ------------------------------------------------------------------ models
def random_shift(images, pad=4):
    n, c, h, w = images.shape
    padded = F.pad(images, (pad, pad, pad, pad), mode='replicate')
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=images.device)[:h]
    base = torch.stack(torch.meshgrid(arange, arange, indexing='xy'), -1)[None].repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, (n, 1, 1, 2), device=images.device).float() * 2.0 / (h + 2 * pad)
    return F.grid_sample(padded, base + shift, padding_mode='zeros', align_corners=False)


class Policy(GaussianMixin, Model):
    """Images + proprioception only (deployable)."""

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-5, max_log_std=1)
        self.encoder = Encoder()
        self.net = nn.Sequential(nn.Linear(64 + N_PROPRIO, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, self.num_actions), nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), -1.0))

    def compute(self, inputs, role):
        states = inputs['states']
        images = states[:, :N_IMG].reshape(-1, 6, IMG, IMG) / 255.0
        if self.training:
            images = random_shift(images)
        proprio = states[:, N_IMG:N_IMG + N_PROPRIO]
        return self.net(torch.cat([self.encoder(images), proprio], 1)), self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    """Sim state + proprioception + action (training only)."""

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(nn.Linear(N_VEC + self.num_actions, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs['states'][:, N_IMG:], inputs['taken_actions']], 1)), {}


# ------------------------------------------------------------------ agent
class SACfD(SAC):
    """skrl SAC with images stored as uint8, a separate demonstration buffer (half of every batch, RLPD) and a
    behavior-cloning term on the demonstration half of the actor loss."""

    NAMES = ['img', 'vec', 'actions', 'rewards', 'next_img', 'next_vec', 'terminated', 'truncated']

    def __init__(self, *args, demo_memory=None, bc_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_memory = demo_memory
        self.bc_weight = bc_weight

    def init(self, trainer_cfg=None):
        Agent.init(self, trainer_cfg=trainer_cfg)
        self.set_mode('eval')
        for memory in (self.memory, self.demo_memory):
            if memory.tensors:
                continue
            memory.create_tensor(name='img', size=N_IMG, dtype=torch.uint8)
            memory.create_tensor(name='next_img', size=N_IMG, dtype=torch.uint8)
            memory.create_tensor(name='vec', size=N_VEC, dtype=torch.float32)
            memory.create_tensor(name='next_vec', size=N_VEC, dtype=torch.float32)
            memory.create_tensor(name='actions', size=self.action_space, dtype=torch.float32)
            memory.create_tensor(name='rewards', size=1, dtype=torch.float32)
            memory.create_tensor(name='terminated', size=1, dtype=torch.bool)
            memory.create_tensor(name='truncated', size=1, dtype=torch.bool)

    @staticmethod
    def _split(states):
        return states[:, :N_IMG].round().to(torch.uint8), states[:, N_IMG:]

    def _store(self, memory, states, actions, rewards, next_states, terminated, truncated):
        img, vec = self._split(states)
        next_img, next_vec = self._split(next_states)
        memory.add_samples(img=img, vec=vec, actions=actions, rewards=rewards, next_img=next_img, next_vec=next_vec,
                           terminated=terminated, truncated=truncated)

    def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
        Agent.record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep,
                                timesteps)
        self._store(self.memory, states, actions, rewards, next_states, terminated, truncated)

    def add_demo(self, states, actions, rewards, next_states, terminated, truncated):
        """Many transitions into the (num_envs = 1) demo memory. Written directly: skrl 1.4.3's multi-sample
        add_samples advances the write index once per tensor, so len() then counts unwritten rows."""
        img, vec = self._split(states)
        next_img, next_vec = self._split(next_states)
        memory = self.demo_memory
        count = min(len(img), memory.memory_size - memory.memory_index)
        for name, value in (('img', img), ('vec', vec), ('actions', actions), ('rewards', rewards), ('next_img', next_img),
                            ('next_vec', next_vec), ('terminated', terminated), ('truncated', truncated)):
            memory.tensors[name][memory.memory_index:memory.memory_index + count, 0].copy_(value[:count])
        memory.memory_index += count
        if memory.memory_index >= memory.memory_size:
            memory.memory_index, memory.filled = 0, True

    def _sample(self, memory, size):
        img, vec, actions, rewards, next_img, next_vec, terminated, truncated = memory.sample(
            names=self.NAMES, batch_size=size)[0]
        return (torch.cat([img.float(), vec], 1), actions, rewards, torch.cat([next_img.float(), next_vec], 1),
                terminated, truncated)

    def bc_loss(self, states, actions):
        _, _, outputs = self.policy.act({'states': states}, role='policy')
        return F.mse_loss(outputs['mean_actions'], actions)

    def _update(self, timestep, timesteps):
        half = self._batch_size // 2
        for _ in range(self._gradient_steps):
            online = self._sample(self.memory, half)
            demo = self._sample(self.demo_memory, half)
            states, actions, rewards, next_states, terminated, truncated = [torch.cat(x) for x in zip(online, demo)]
            with torch.no_grad():
                next_actions, next_log_prob, _ = self.policy.act({'states': next_states}, role='policy')
                q1, _, _ = self.target_critic_1.act({'states': next_states, 'taken_actions': next_actions},
                                                     role='target_critic_1')
                q2, _, _ = self.target_critic_2.act({'states': next_states, 'taken_actions': next_actions},
                                                     role='target_critic_2')
                target = rewards + self._discount_factor * (terminated | truncated).logical_not() * (
                    torch.min(q1, q2) - self._entropy_coefficient * next_log_prob)
            c1, _, _ = self.critic_1.act({'states': states, 'taken_actions': actions}, role='critic_1')
            c2, _, _ = self.critic_2.act({'states': states, 'taken_actions': actions}, role='critic_2')
            critic_loss = (F.mse_loss(c1, target) + F.mse_loss(c2, target)) / 2
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), 10.0)
            self.critic_optimizer.step()

            new_actions, log_prob, _ = self.policy.act({'states': states}, role='policy')
            q1, _, _ = self.critic_1.act({'states': states, 'taken_actions': new_actions}, role='critic_1')
            q2, _, _ = self.critic_2.act({'states': states, 'taken_actions': new_actions}, role='critic_2')
            bc = self.bc_loss(demo[0], demo[1])
            policy_loss = (self._entropy_coefficient * log_prob - torch.min(q1, q2)).mean() + self.bc_weight * bc
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.policy_optimizer.step()

            if self._learn_entropy:
                entropy_loss = -(self.log_entropy_coefficient * (log_prob + self._target_entropy).detach()).mean()
                self.entropy_optimizer.zero_grad()
                entropy_loss.backward()
                self.entropy_optimizer.step()
                self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())

            self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)
        self.track_data('Loss / Critic loss', critic_loss.item())
        self.track_data('Loss / Policy loss', policy_loss.item())
        self.track_data('Loss / BC loss', bc.item())
        self.track_data('Q-network / Q1 (mean)', c1.mean().item())
        self.track_data('Coefficient / Entropy coefficient', float(self._entropy_coefficient))


def make_agent(env, out, memory_size, demo_size, gradient_steps, bc_weight):
    device = env.device
    models = {'policy': Policy(env.observation_space, env.action_space, device)}
    for name in ('critic_1', 'critic_2', 'target_critic_1', 'target_critic_2'):
        models[name] = Critic(env.observation_space, env.action_space, device)
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg.update({'gradient_steps': gradient_steps, 'batch_size': 256, 'discount_factor': 0.98, 'polyak': 0.005,
                'actor_learning_rate': 3e-4, 'critic_learning_rate': 3e-4, 'learn_entropy': True,
                'initial_entropy_value': 0.05, 'learning_starts': 10, 'random_timesteps': 0})
    cfg['experiment'] = dict(cfg['experiment'], directory=str(out), experiment_name='sac', write_interval=100,
                             checkpoint_interval=1000)
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)
    demo_memory = RandomMemory(memory_size=demo_size, num_envs=1, device=device)
    return SACfD(models=models, memory=memory, demo_memory=demo_memory, bc_weight=bc_weight, cfg=cfg,
                 observation_space=env.observation_space, action_space=env.action_space, device=device)


# ------------------------------------------------------------------ phases
def collect_demos(env, agent, count):
    """Runs the scripted hand through the environment; keeps the transitions of successful episodes."""
    env.scripted = [None] * env.n
    obs, _ = env.reset()
    episodes = [[] for _ in range(env.n)]
    saved, start = 0, time.time()
    while saved < count:
        actions = env.scripted_actions()
        next_obs, reward, terminated, truncated, info = env.step(actions)
        for i in range(env.n):
            # a finished episode's next observation is already the reset one; terminal transitions don't bootstrap
            episodes[i].append((obs[i, :N_IMG].byte(), obs[i, N_IMG:], actions[i], reward[i], next_obs[i, :N_IMG].byte(),
                                next_obs[i, N_IMG:], terminated[i], truncated[i]))
            if terminated[i] or truncated[i]:
                if info['success'][i] and saved < count:
                    img, vec, acts, rews, next_img, next_vec, terms, truncs = [torch.stack(x) for x in zip(*episodes[i])]
                    agent.add_demo(torch.cat([img.float(), vec], 1), acts, rews, torch.cat([next_img.float(), next_vec], 1),
                                   terms, truncs)
                    saved += 1
                episodes[i] = []
        obs = next_obs
    attempts = len(env.finished)
    print('[%s] demonstrations: %d successful of %d scripted episodes in %.0f s' % (
        time.strftime('%H:%M:%S'), saved, attempts, time.time() - start), flush=True)
    env.scripted = None
    env.finished = []


def pretrain_bc(agent, steps):
    agent.set_mode('train')
    for k in range(steps):
        states, actions, *_ = agent._sample(agent.demo_memory, 256)
        loss = agent.bc_loss(states, actions)
        agent.policy_optimizer.zero_grad()
        loss.backward()
        agent.policy_optimizer.step()
        if k % 500 == 0 or k == steps - 1:
            print('[%s] BC %d / %d: loss %.4f' % (time.strftime('%H:%M:%S'), k, steps, loss.item()), flush=True)
    agent.set_mode('eval')


def evaluate(env, agent, episodes, out, videos=8, label='eval'):
    """Deterministic policy (mean action) on random cup placements; success rate + policy-input videos."""
    agent.set_mode('eval')
    obs, _ = env.reset()
    env.finished = []
    frames = [[] for _ in range(env.n)]
    saved_videos = 0
    while len(env.finished) < episodes:
        with torch.no_grad():
            _, _, outputs = agent.policy.act({'states': obs}, role='policy')
        before = len(env.finished)
        imgs = obs[:, :N_IMG].reshape(-1, 6, IMG, IMG).byte().cpu().numpy()
        obs, _, terminated, truncated, _ = env.step(outputs['mean_actions'])
        for i in range(env.n):
            frames[i].append(np.hstack([imgs[i, :3].transpose(1, 2, 0), imgs[i, 3:].transpose(1, 2, 0)]))
        done = (terminated | truncated)[:, 0].cpu().numpy()
        for j, i in enumerate(np.nonzero(done)[0]):
            if saved_videos < videos:
                success = env.finished[before + j][0]
                path = out / ('%s_%02d_%s.mp4' % (label, saved_videos, 'success' if success else 'fail'))
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (4 * IMG * 2, 4 * IMG))
                for f in frames[i]:
                    writer.write(cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), (4 * IMG * 2, 4 * IMG),
                                            interpolation=cv2.INTER_NEAREST))
                writer.release()
                saved_videos += 1
            frames[i] = []
    results = env.finished[:episodes]
    rate = float(np.mean([r[0] for r in results]))
    (out / ('%s.json' % label)).write_text(json.dumps({'episodes': len(results), 'success_rate': rate,
                                                       'results': results}, indent=1))
    print('[%s] %s: success %d / %d (%.0f%%)' % (time.strftime('%H:%M:%S'), label, sum(r[0] for r in results),
                                                len(results), 100 * rate), flush=True)
    env.finished = []
    return rate


def run(api, args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = CupEnv(api)
    agent = make_agent(env, out, memory_size=args.sac_memory, demo_size=args.sac_demo_memory,
                       gradient_steps=args.sac_gradient_steps, bc_weight=args.sac_bc_weight)
    agent.init()
    if args.sac_eval:
        agent.policy.load_state_dict(torch.load(args.sac_eval, map_location=env.device))
        evaluate(env, agent, args.sac_eval_episodes, out, videos=16, label='eval_' + Path(args.sac_eval).stem)
        return
    collect_demos(env, agent, args.sac_demos)
    if args.sac_init:  # continue from a checkpoint of an earlier run (models, optimizers, entropy)
        agent.load(args.sac_init)
        print('[%s] loaded %s' % (time.strftime('%H:%M:%S'), args.sac_init), flush=True)
    else:
        pretrain_bc(agent, args.sac_bc_steps)
    torch.save(agent.policy.state_dict(), out / 'policy_bc.pt')
    evaluate(env, agent, args.sac_eval_episodes, out, label='eval_bc')
    trainer = SequentialTrainer(cfg={'timesteps': args.sac_steps, 'headless': True, 'disable_progressbar': True,
                                     'close_environment_at_exit': False}, env=env, agents=agent)
    trainer.train()
    torch.save(agent.policy.state_dict(), out / 'policy_final.pt')
    evaluate(env, agent, args.sac_eval_episodes, out, label='eval_final')
    print('SAC_DONE', flush=True)
