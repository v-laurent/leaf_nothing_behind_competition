import os
import pickle
    
import torch
from torch.nn import L1Loss, MSELoss
from torch.nn.functional import mse_loss

from torch.optim import Adam

from .data import get_loader
from .model import get_model


def train(config, tracker):
    """ Uses the config to load the data and a model, then trains the model and logs the results using the tracker. """

    print("Starting training.")
    train_dataloader, val_dataloader = get_loader(config)
    number_of_batches = config.number_of_batches if len(train_dataloader) > config.number_of_batches > -1 else len(train_dataloader)
    model = get_model(config)
    loss_function = MSELoss(reduction="none")
    train_losses = []
    val_losses = []
    
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
            val_loss = evaluate_model(model, val_dataloader)
        model.train() 
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        tracker.log_scalar("loss", total_loss := train_loss, step=epoch)
        print(f"Epoch {epoch+1}/{config.number_of_epochs} : train loss = {train_loss} : val loss = {val_loss}.")
        torch.save(model.state_dict(), os.path.join(config.save_weights_under, f"epoch_{epoch}"))
        
    write_losses(config, train_losses, val_losses)
    
def evaluate_model(model, dataloader):
    losses = []
    for data in dataloader:
        X,y = data["X"], data["y"]
        masks = data["mask"]
        outputs = model(X)
        loss = torch.mean(masks * mse_loss(outputs, y, reduction='none'))
        losses.append(loss.item())
            
    return sum(losses) / len(losses)

def write_losses(config, train_losses, val_losses):
    with open(os.path.join(config.experiment_logs, "train_losses"), "wb") as f:   #Pickling
        pickle.dump(train_losses, f)
        
    with open(os.path.join(config.experiment_logs, "val_losses"), "wb") as f:   #Pickling
        pickle.dump(val_losses, f)