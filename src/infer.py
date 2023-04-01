import os
import pickle

import numpy as np

from .data import get_loader
from .model import get_model


def infer(config):
    """ Saves the prediction results in numpy format in the folder config.save_infers_under """
    loader = get_loader(config)
    number_of_batches = config.number_of_batches if len(loader) > config.number_of_batches > -1 else len(loader)
    model = get_model(config)
    results = {"outputs": [], "paths": []}

    for i, data in enumerate(loader):
        if i == number_of_batches:
            break
        results["paths"] += list(data["paths"][-1])
        results["outputs"].append(model(data).detach().cpu().numpy())
        if i % 10 == 9 or i+1 == number_of_batches:
            print(f"Performed batch {i+1}/{number_of_batches}")

    results["outputs"] = np.concatenate(results["outputs"], axis=0)
    with open(os.path.join(config.save_infers_under, "results.pickle"), 'wb') as file:
        pickle.dump(results, file)
