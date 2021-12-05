import os
import sys

from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

sys.path.append(os.getcwd())

from lib.custom_env import CustomEnv
from lib.summary_writer import SummaryWriterCallback

MODEL_PATH = "data/model/model"
RESULTS_DIR = "./tensorboard/"

N_TRAIN_EPISODES = 1000
N_EVAL_EPISODES = 5


def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # 環境を定義
    custom_envs = make_vec_env(CustomEnv, n_envs=4)
    N_STEPS = custom_envs.envs[0].max_episode_timesteps

    # dqnのmodelを作成
    model = PPO("MlpPolicy", custom_envs, learning_rate=0.0005,
                n_steps=32, batch_size=64,
                n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, clip_range_vf=None,
                ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                use_sde=False, sde_sample_freq=- 1,
                target_kl=None, tensorboard_log=RESULTS_DIR,
                create_eval_env=False, policy_kwargs=None,
                verbose=1, seed=0, device='auto', _init_setup_model=True)

    # custom_env = CustomEnv()
    # model = DDPG("MlpPolicy", env=custom_env, buffer_size=10000, batch_size=32, verbose=1, seed=0)
    # model = SAC("MlpPolicy",
    #             env=custom_env,
    #             learning_rate=3e-4,
    #             buffer_size=50000,
    #             batch_size=64,
    #             ent_coef="auto",
    #             train_freq=1,
    #             gradient_steps=1,
    #             gamma=0.99,
    #             tau=0.01,
    #             learning_starts=1000,
    #             policy_kwargs=dict(net_arch=[64, 64]),
    #             verbose=1,
    #             seed=0)

    model.learn(total_timesteps=int(N_TRAIN_EPISODES * N_STEPS),
                callback=SummaryWriterCallback())
    # model.learn(total_timesteps=int(N_TRAIN_EPISODES * N_STEPS))
    model.save(MODEL_PATH)

    del model, custom_envs

    custom_envs = make_vec_env(CustomEnv, n_envs=4)
    model = PPO.load(MODEL_PATH, env=custom_envs)
    rewards, steps = evaluate_policy(model, custom_envs,
                                     N_EVAL_EPISODES, return_episode_rewards=True)
    # custom_env = CustomEnv()
    # model = DDPG.load(MODEL_PATH, env=custom_env)
    # model = SAC.load(MODEL_PATH, env=custom_env)
    # rewards, steps = evaluate_policy(model, custom_env,
    #                                  N_EVAL_EPISODES, return_episode_rewards=True)
    print(f"Rewards (Evaluation): {str(rewards)}")


if __name__ == "__main__":
    main()
