import matplotlib.pyplot as plt
import numpy as np

from .data import get_dataset
from .model import get_model


def infer_on_train(config):
    assert config.weights_path is not None, "The model needs to be trained before"
    
    dataset = get_dataset(config)
    model = get_model(config)
    
    keep_inferring =True
    print("Starting inference")
    while keep_inferring:
        index = np.random.choice(np.arange(len(dataset)))
        sample = dataset[index]
        
        plt.subplot(221)
        plt.imshow(sample["X"][0,:,:].detach().numpy().T)
        plt.axis('off')
        plt.title("S1")
        
        plt.subplot(222)
        plt.imshow(sample["X"][1,:,:].detach().numpy().T)
        plt.axis('off')
        plt.title("S1")
        
        plt.subplot(223)
        plt.imshow(sample["y"].detach().numpy().T)
        plt.axis('off')
        plt.title("ground truth")
        
        plt.subplot(224)
        output = model(sample["X"])
        plt.imshow(output.detach().numpy().T)
        plt.axis('off')
        plt.title("predicted")
        
        plt.show()
        keep_inferring = input("Continue ?(Y/n) \n  >>>>>") != "n"
        
    print("Inferring done")