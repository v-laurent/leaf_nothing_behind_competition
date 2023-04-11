import os
import pickle
    
import torch
from torch.nn import MSELoss
from torch.nn.functional import mse_loss

from torch.optim import Adam

from .data import get_loader
from .model import get_model


def train_S1toS2(config, tracker):
    """ Uses the config to load the data and a model, then trains the model and logs the results using the tracker. """

    print("Starting training.")
    train_dataloader, val_dataloader = get_loader(config)
    number_of_batches = config.number_of_batches if len(train_dataloader) > config.number_of_batches > -1 else len(train_dataloader)
    model = get_model(config)
    loss_function = MSELoss(reduction="none")
    
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.number_of_epochs):
        losses = []

        for i, data in enumerate(train_dataloader):
            if i == number_of_batches:
                break
            X,y = data["X"], data["y"]
            masks = data["mask"]
            optimizer.zero_grad()
            outputs = model(X)
            loss = torch.mean(masks * loss_function(outputs, y))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i % 100 == 99 or i+1 == number_of_batches:
                print(f"Performed batch {i+1}/{number_of_batches}")

        train_loss = sum(losses)/len(losses)
        
        model.eval()
        with torch.no_grad():
            val_loss = evaluate_S1toS2model(model, val_dataloader)
        model.train() 
        
        add_element_to_file(config, train_loss, "train_losses.pkl")
        add_element_to_file(config, val_loss, "val_losses.pkl")
        
        tracker.log_scalar("loss", total_loss := train_loss, step=epoch)
        print(f"Epoch {epoch+1}/{config.number_of_epochs} : train loss = {train_loss} : val loss = {val_loss}.")
        torch.save(model.state_dict(), os.path.join(config.save_weights_under, f"epoch_{epoch}"))
        
def train_Unet(config, tracker):
    
    print("Starting training.")
    train_dataloader, val_dataloader = get_loader(config)
    number_of_batches = config.number_of_batches if len(train_dataloader) > config.number_of_batches > -1 else len(train_dataloader)
    model = get_model(config)
    loss_function = MSELoss(reduction="none")
    
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.number_of_epochs):
        losses = []

        for i, data in enumerate(train_dataloader):
            if i == number_of_batches:
                break
            _, S1_t0_data, S2_and_Mask_t2, S2_and_Mask_t1, y, y_mask = data.values()
            optimizer.zero_grad()
            outputs = model(S1_t0_data, S2_and_Mask_t2, S2_and_Mask_t1)
            mask_loss = torch.mul(S2_and_Mask_t2[:,1,:,:],S2_and_Mask_t1[:,1,:,:]).unsqueeze(1) + 1
            loss = torch.mean(config.lamda * mask_loss * loss_function(outputs, y))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i % 100 == 99 or i+1 == number_of_batches:
                print(f"Performed batch {i+1}/{number_of_batches}")
       
        train_loss = sum(losses)/len(losses)
        
        model.eval()
        with torch.no_grad():
            val_loss = evaluate_Unetmodel(model, val_dataloader, config)
        model.train() 
        
        add_element_to_file(config, train_loss, "train_losses.pkl")
        add_element_to_file(config, val_loss, "val_losses.pkl")
        
        tracker.log_scalar("loss", total_loss := train_loss, step=epoch)
        print(f"Epoch {epoch+1}/{config.number_of_epochs} : train loss = {train_loss} : val loss = {val_loss}.")
        torch.save(model.state_dict(), os.path.join(config.save_weights_under, f"epoch_{epoch}"))
        
    
def evaluate_S1toS2model(model, dataloader):
    losses = []
    for data in dataloader:
        X,y = data["X"], data["y"]
        masks = data["mask"]
        outputs = model(X)
        loss = torch.mean(masks * mse_loss(outputs, y, reduction='none'))
        losses.append(loss.item())
            
    return sum(losses) / len(losses)
        
def evaluate_Unetmodel(model, dataloader, config):
    losses = []
    for data in dataloader:
        _, S1_t0_data, S2_and_Mask_t2, S2_and_Mask_t1, y, y_mask = data.values()
        outputs = model(S1_t0_data, S2_and_Mask_t2, S2_and_Mask_t1)
        loss = torch.mean(config.lamda*(2-y_mask) * mse_loss(outputs, y, reduction='none'))
        losses.append(loss.item())
            
    return sum(losses) / len(losses)

def add_element_to_file(config, el, file_name):
    path = os.path.join(config.experiment_logs, file_name)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        data.append(el)
    else:
        data = [el]
    
    with open(path, 'wb') as f:
        pickle.dump(data, f)