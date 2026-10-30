"""Deep Q-Network (Mnih et al., 2015) with Stable-Baselines3 on Gymnasium environments.

  python models/zoo/dqn.py train    [--env CartPole-v1] [--steps 60000]
  python models/zoo/dqn.py play     [--episodes 5]
  python models/zoo/dqn.py selftest           (CartPole-v1, ships with gymnasium, no downloads)

DQN learns Q(s, a) with a neural network, an experience-replay buffer, a periodically updated target
network and epsilon-greedy exploration: Q <- r + gamma * max_a' Q_target(s', a').
Hyper-parameters for CartPole follow the rl-baselines3-zoo tuned settings. The trained agent is compared
with a random policy on the same environment.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import common as C  # noqa: E402
import _data as D  # noqa: E402

ZOO_CARTPOLE = dict(learning_rate=2.3e-3, batch_size=64, buffer_size=100000, learning_starts=1000, gamma=0.99, target_update_interval=10, train_freq=256,
                    gradient_steps=128, exploration_fraction=0.16, exploration_final_eps=0.04, policy_kwargs=dict(net_arch=[256, 256]))


def random_return(env_id, episodes, seed):
    import gymnasium as gym
    env = gym.make(env_id); rs = []
    for i in range(episodes):
        env.reset(seed=seed + i); done, tot = False, 0.0
        while not done:
            _, r, term, trunc, _ = env.step(env.action_space.sample()); tot += r; done = term or trunc
        rs.append(tot)
    return float(np.mean(rs))


def train(args):
    import gymnasium as gym
    from stable_baselines3 import DQN
    from stable_baselines3.common.evaluation import evaluate_policy
    C.seed_all(args.seed)
    env = gym.make(args.env)
    hp = dict(ZOO_CARTPOLE) if args.env.startswith("CartPole") else {}
    model = DQN("MlpPolicy", env, seed=args.seed, verbose=0, device="cpu", **hp)   # small MLP policy: CPU is fastest
    C.log("DQN on %s · %d steps · %s" % (args.env, args.steps, "zoo hyper-parameters" if hp else "SB3 defaults"))
    model.learn(total_timesteps=args.steps, progress_bar=False)
    mean, std = evaluate_policy(model, gym.make(args.env), n_eval_episodes=args.eval_episodes, deterministic=True)
    rnd = random_return(args.env, args.eval_episodes, args.seed)
    out = D.out_dir("dqn"); model.save(os.path.join(out, "dqn_%s" % args.env))
    C.save_json(os.path.join(out, "report.json"), {"env": args.env, "steps": args.steps, "mean_return": float(mean), "std_return": float(std), "random_policy_return": rnd})
    C.log("evaluation return %.1f ± %.1f (random policy %.1f) · saved to %s" % (mean, std, rnd, out))
    return float(mean), rnd


def play(args):
    import gymnasium as gym
    from stable_baselines3 import DQN
    model = DQN.load(os.path.join(D.out_dir("dqn"), "dqn_%s" % args.env), device="cpu")
    env = gym.make(args.env, render_mode="human" if args.render else None)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep); done, tot = False, 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True); obs, r, term, trunc, _ = env.step(a); tot += r; done = term or trunc
        C.log("episode %d return %.0f" % (ep + 1, tot))


def selftest(args):
    mean, rnd = train(argparse.Namespace(env="CartPole-v1", steps=50000, eval_episodes=10, seed=0))
    C.check(mean > max(100, 3 * rnd), "trained agent return %.1f vs random %.1f (needs > max(100, 3x random))" % (mean, rnd))
    C.log("DQN SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--env", default="CartPole-v1"); t.add_argument("--steps", type=int, default=60000)
    t.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=20); t.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("play"); p.add_argument("--env", default="CartPole-v1"); p.add_argument("--episodes", type=int, default=5); p.add_argument("--render", action="store_true")
    s = sub.add_parser("selftest"); s.add_argument("--device")
    a = ap.parse_args(); {"train": train, "play": play, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
