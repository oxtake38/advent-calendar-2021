import matplotlib.pyplot as plt

from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.callbacks import BaseCallback


class SummaryWriterCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(SummaryWriterCallback, self).__init__(verbose)

    def _on_training_start(self):
        self._log_freq = 1  # log every 1000 calls
        self._fig_freq_episode = 25

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))
        self.ep = 0

        self.states = {}
        self.actions = {}
        self.rewards = {}
        self.targets = {}

    def _on_step(self) -> bool:
        """This method will be called by the model after each call to ``env.step()``.
        """
        n_env = 0

        ep = f"Ep. {self.ep}"

        if self.locals['dones'][n_env]:
            self.states[ep] = self.locals["infos"][n_env]["states"]
            add_figure(self.states, self.tb_formatter, n_env, self._log_freq, self._fig_freq_episode)
            self.actions[ep] = self.locals["infos"][n_env]["actions"]
            add_figure(self.actions, self.tb_formatter, n_env, self._log_freq, self._fig_freq_episode)
            self.rewards[ep] = self.locals["infos"][n_env]["rewards"]
            add_figure(self.rewards, self.tb_formatter, n_env, self._log_freq, self._fig_freq_episode)
            self.targets[ep] = self.locals["infos"][n_env]["targets"]
            add_figure(self.targets, self.tb_formatter, n_env, self._log_freq, self._fig_freq_episode)
            self.tb_formatter.writer.flush()
            self.ep += 1
        return True


def add_figure(df, tb_formatter, n_env, log_freq, fig_freq_episode):
    for col in list(df.values())[0].columns:
        fig = plt.figure(figsize=(12, 10))
        for key in list(df.keys())[::fig_freq_episode]:
            df[key][col][::log_freq].plot(alpha=0.5)
        plt.legend(list(df.keys())[::fig_freq_episode])
        tb_formatter.writer.add_figure(f'user/env-{n_env}_{col}', fig)
