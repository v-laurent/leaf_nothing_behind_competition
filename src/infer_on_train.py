import matplotlib.pyplot as plt
import numpy as np

from .data import get_dataset
from .model import get_model
from torch.nn.functional import mse_loss


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
    
def infer_on_train_unet(config):
    assert config.weights_path is not None, "The model needs to be trained before"
    
    dataset = get_dataset(config)
    model = get_model(config)
        
    keep_inferring =True
    print("Starting inference")
    while keep_inferring:
        index = np.random.choice(np.arange(len(dataset)))
        sample = dataset[index]
        _, S1_t0_data, S2_and_Mask_t2, S2_and_Mask_t1, y, y_mask = sample.values()
        outputs = model(S1_t0_data.unsqueeze(0), S2_and_Mask_t2.unsqueeze(0), S2_and_Mask_t1.unsqueeze(0))
    
        plt.subplot(331)
        plt.imshow(S1_t0_data[0,:,:].detach().numpy().T)
        plt.axis('off')
        plt.title("S1")
        
        plt.subplot(332)
        plt.imshow(S1_t0_data[1,:,:].detach().numpy().T)
        plt.axis('off')
        plt.title("S1")
        
        plt.subplot(333)
        plt.imshow(y[0,:,:].detach().numpy().T)
        plt.axis('off')
        plt.title("ground truth")
        
        plt.subplot(334)
        plt.imshow(S2_and_Mask_t2[0,:,:].detach().numpy().T)
        plt.axis('off')
        plt.title("S2 T2")
        
        plt.subplot(335)
        plt.imshow(S2_and_Mask_t1[0,:,:].detach().numpy().T)
        plt.axis('off')
        plt.title("S2 T1")
        
        plt.subplot(336)
        plt.imshow(outputs[0,0,:,:].detach().numpy().T)        
        plt.axis('off')
        plt.title("predicted")

        plt.subplot(337)
        plt.imshow(S2_and_Mask_t2[1,:,:].detach().numpy().T)
        plt.axis('off')
        plt.title("Mask T2")
        
        plt.subplot(338)
        plt.imshow(S2_and_Mask_t1[1,:,:].detach().numpy().T)
        plt.axis('off')
        plt.title("Mask T1")
        
        plt.subplot(339)
        plt.imshow(y_mask[0,:,:].detach().numpy().T)
        plt.axis('off')
        plt.title("Mask ground truth")
         
        loss = mse_loss(outputs, y)
        plt.suptitle(f"Loss = {loss}")
        plt.show()
        
        keep_inferring = input("Continue ?(Y/n) \n  >>>>>") != "n"
        
    print("Inferring done")