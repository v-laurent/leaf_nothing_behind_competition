import matplotlib.pyplot as plt
import numpy as np

from .data import get_dataset
from .model import get_model
from torch.nn.functional import mse_loss
from pytorch_msssim import ssim


def infer_on_train(config):
    assert config.weights_path is not None, "The model needs to be trained before"
    
    dataset = get_dataset(config)
    model = get_model(config)
    
    keep_inferring =True
    print("Starting inference")
    while keep_inferring:
        index = np.random.choice(np.arange(len(dataset)))
        sample = dataset[index]
        
        f = plt.figure(figsize=(7,8))
        a1 = f.add_subplot(321)
        a1.imshow(sample["X"][0,:,:].detach().numpy().T)
        a1.axis('off')
        
        
        a2 = f.add_subplot(322)
        a2.imshow(sample["X"][1,:,:].detach().numpy().T)
        a2.axis('off')
       
        
        a3 = f.add_subplot(323)
        a3.imshow(sample["y"].detach().numpy().T)
        a3.axis('off')
        
        mask =  sample["mask"].detach().numpy().T
        a4 = f.add_subplot(324)
        a4.imshow(sample["mask"].detach().numpy().T)
        a4.axis('off')
        
        
        a5 = f.add_subplot(325)
        output = model(sample["X"])
        a5.imshow(output.detach().numpy().T)
        a5.axis('off')
        
        print(256*256 - np.sum(mask))
        plt.show()
        # keep_inferring = input("Continue ?(Y/n) \n  >>>>>") != "n"
        
    print("Inferring done")
    
def infer_on_train_unet(config):
    assert config.weights_path is not None, "The model needs to be trained before"
    
    dataset = get_dataset(config)
    model = get_model(config)
        
    keep_inferring =True
    i = 0
    print("Starting inference")
    while keep_inferring:
        i+=1
        index = np.random.choice(np.arange(len(dataset)))
        sample = dataset[index]
        _, S1_t0_data, S2_and_Mask_t2, S2_and_Mask_t1, y, y_mask = sample.values()
        outputs = model(S1_t0_data.unsqueeze(0), S2_and_Mask_t2.unsqueeze(0), S2_and_Mask_t1.unsqueeze(0))

        y_mask = y_mask[0,:,:].detach().numpy().T
        inpurity = (256*256 - np.sum(y_mask))/(256*256)

        maskt2 = S2_and_Mask_t2[1,:,:].detach().numpy().T
        inpurity2 = (256*256 - np.sum(maskt2))/(256*256)

        maskt1 = S2_and_Mask_t1[1,:,:].detach().numpy().T
        inpurity1 = (256*256 - np.sum(maskt1))/(256*256)
        

        print(i, inpurity)
        if(inpurity > 0.01 or inpurity2>0.01 or inpurity1>0.01):

            plt.figure(figsize=(10, 8))


            plt.subplot(333)
            plt.imshow(S1_t0_data[0,:,:].detach().numpy().T)
            plt.axis('off')
            plt.title("S1")
            
            plt.subplot(336)
            plt.imshow(S1_t0_data[1,:,:].detach().numpy().T)
            plt.axis('off')
            plt.title("S1")
            
            plt.subplot(338)
            plt.imshow(y[0,:,:].detach().numpy().T)
            plt.axis('off')
            plt.title("ground truth")
            
            plt.subplot(331)
            plt.imshow(S2_and_Mask_t2[0,:,:].detach().numpy().T)
            plt.axis('off')
            plt.title("S2 T2")
            
            plt.subplot(332)
            plt.imshow(S2_and_Mask_t1[0,:,:].detach().numpy().T)
            plt.axis('off')
            plt.title("S2 T1")
            
            plt.subplot(339)
            plt.imshow(outputs[0,0,:,:].detach().numpy().T)        
            plt.axis('off')
            plt.title("predicted")

            plt.subplot(334)
            
            plt.imshow(maskt2)
            plt.axis('off')
            plt.title(f"Mask T2 - {inpurity2:.3f}")
            
            plt.subplot(335)
            plt.imshow(S2_and_Mask_t1[1,:,:].detach().numpy().T)
            plt.axis('off')
            plt.title(f"Mask T1 - {inpurity1:.3f}")
            
            plt.subplot(337)
            plt.imshow(y_mask)
            plt.axis('off')
            plt.title(f"Mask ground truth - {inpurity:.3f}")
            

            ssim_value = ssim(outputs, y.unsqueeze(0), data_range=outputs[0].max() - y.min()).item()

            loss = mse_loss(outputs[0], y)
            plt.suptitle(f"Loss = {loss} - SSMI = {ssim_value}")
            plt.show()
            

       
        # keep_inferring = input("Continue ?(Y/n) \n  >>>>>") != "n"
        
    print("Inferring done")