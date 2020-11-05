import argparse
import json
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt


# imports the model in model.py by name
from model import Classifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier()

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model

# Gets training data in batches from the train.csv file
def _get_train_data_loader(batch_size,training_dir,class_weight_0,class_weight_1):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    train_y = torch.from_numpy(train_data[[3197]].values).float().squeeze()
    train_x = torch.from_numpy(train_data.drop([3197], axis=1).values).float()
    
    samples_weight = ([class_weight_1/37]*37) + ([class_weight_0/(5050)]*5050) 
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    
    train_ds = torch.utils.data.TensorDataset(train_x, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size, num_workers=0, sampler=sampler)

def _get_val_data(training_dir):
    print("Get validation data loader.")

    valid_data = pd.read_csv(os.path.join(training_dir, "test.csv"), header=None, names=None)

    valid_y = torch.from_numpy(valid_data[[3197]].values).float().squeeze()
    valid_x = torch.from_numpy(valid_data.drop([3197], axis=1).values).float()
    

    return valid_x,valid_y


def accuracy(y_pred,y):
    assert len(y_pred)==len(y)
    
    c=0.0
    for i in range(0,len(y)):
        if(int(torch.round(y_pred[i])) == y[i]):
            c=c+1
    c=c/len(y)
    return c

def weighted_binary_cross_entropy(output, target, weights=None):
        
#     if weights is not None:
    assert len(weights) == 2
        
    loss = weights[1] * (target * torch.log(output)) + \
          weights[0] * ((1 - target) * torch.log(1 - output))
#     else:
#         loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

# Provided training function
def train(model, train_loader, epochs, optimizer, device, val_data,criterion):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    val_data     - The data that should be used during validation.
    criterion    - The loss function used for training.
    
    Returns:
    epoch_train_loss                  - array containing the values of training loss in every epoch
    epoch_train_accuracy        - array containing the values of training accuracy in every epoch
    epoch_validation_accuracy   - array containing the values of validation accuracy in every epoch
    epoch_validation_loss       - array containing the values of validation loss in every epoch
    """
            
    epoch_train_loss=[]
    epoch_train_accuracy=[]
    epoch_validation_accuracy=[]
    epoch_validation_loss=[]
    
    for epoch in range(1, epochs + 1):
        
        #train
        model.train() 

        total_loss = 0.
        epoch_acc=0.
        for batch in train_loader:
            # get data
            batch_x, batch_y = batch

            batch_x = batch_x.unsqueeze(1).to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # get predictions from model
            y_pred = model(batch_x)
            
            # perform backprop
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_acc += accuracy(y_pred,batch_y)
            total_loss += loss.data.item()
            
        epoch_train_accuracy.append(epoch_acc/len(train_loader))
        epoch_train_loss.append(total_loss/len(train_loader))
        
        
        
        #validation
        # model.eval()
        
        # valid_accuracy=0.
        # valid_loss=0.
        
        # val_x,val_y = val_data
        # val_x = val_x.unsqueeze(1).to(device)
        # val_y = val_y.to(device)
        
        # with torch.no_grad():
        #     val_pred = model(val_x)
        #     val_loss = criterion(val_pred, val_y)
        #     val_acc = accuracy(val_pred,val_y)
            
        #     epoch_validation_accuracy.append(val_acc)
        #     epoch_validation_loss.append(val_loss)
            
        print("Epoch: {}, Train loss: {}, Train accuracy: {}".format(epoch, total_loss/len(train_loader),epoch_acc/len(train_loader)))
        print("Validation loss: {} Validation accuracy: {}".format(val_loss, val_acc))
        
    return epoch_train_loss,epoch_train_accuracy,epoch_validation_loss,epoch_validation_accuracy


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Training Parameters, given
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--class_weight_0', type=int, default=9, metavar='R0',
                        help='class weight for 0 (default: 9)')
    parser.add_argument('--class_weight_1', type=int, default=1, metavar='R1',
                        help='class weight for 1 (default: 1)')
    
   
    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader =_get_train_data_loader(args.batch_size,args.data_dir,class_weight_0=args.class_weight_0,class_weight_1=args.class_weight_1)
    # Load validation data.
    val_data = _get_val_data(training_dir=args.data_dir)

    model = Classifier().to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.BCELoss()

    # Trains the model (given line of code, which calls the above training function)
    train_hist,train_acc,val_hist,val_acc = train(model, train_loader, args.epochs, optimizer, device,val_data,criterion)
    

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)

        
    # Save the loss history
    hist_path = os.path.join(args.output_data_dir, 'train_loss.txt')
    with open(hist_path, 'w') as f:
        for i in train_hist:
            f.write(str(i)+"\n")
            
    # hist_path = os.path.join(args.output_data_dir, 'val_loss.txt')
    # with open(hist_path, 'w') as f:
    #     for i in val_hist:
    #         f.write(str(i)+"\n")
            
            
#     Save the acc history
    acc_path = os.path.join(args.output_data_dir, 'train_acc.txt')
    with open(acc_path, 'w') as f:
        for i in train_acc:
            f.write(str(i)+"\n")
            
    # acc_path = os.path.join(args.output_data_dir, 'val_acc.txt')
    # with open(acc_path, 'w') as f:
    #     for i in val_acc:
    #         f.write(str(i)+"\n")