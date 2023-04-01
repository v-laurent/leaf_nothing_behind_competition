from yaecs import Experiment
import torch.multiprocessing as mp

from configs.config import BaselineConfig
from src.infer import infer
from src.train import train


def main(config, tracker):

    mp.set_start_method('spawn')

    if config.mode == "train":
        train(config=config, tracker=tracker)

    elif config.mode == "infer":
        infer(config=config)

    else:
        raise ValueError(f"Unknown mode '{config.mode}'. Possible values are 'train' and 'infer'.")


if __name__ == '__main__':
    configuration = BaselineConfig.build_from_argv("paths_on_your_machine", fallback="infer_config")
    print(configuration.details())
    Experiment(main_function=main, config=configuration).run(run_description=configuration.experiment_purpose)
