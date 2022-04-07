import os

import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from model.network import DCN
from dataset.DCN_dataset import DCNDataset

# init data keys, use when training and validation.
data_keys = ["blink","yawn","gaze_x","gaze_y","heart_rate","images"]
# get current device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(epoch, trainloader):
    global loss_values
    print("Epoch:{}".format(epoch))
    # convert model to train mode
    net.train()
    total_loss = []
    for data in tqdm(trainloader, leave=False, total=len(trainloader)):
        inputs=data[0]
        targets=data[1]
        dcn_inputs = {}

        # set the model input
        for idx,key in enumerate(data_keys):
            dcn_inputs[key] = inputs[idx].to(device)

        targets=targets.to(device)
        # clear gradient
        optimizer.zero_grad()

        # forward
        outputs = net(dcn_inputs)
        loss = criterion(outputs, targets)
        # The gradient back propagation of loss is used to update the model parameters
        loss.backward()
        loss_value = loss.item()
        loss_values.append(loss_value)
        total_loss.append(loss_value)

        optimizer.step()

    # record train loss during training
    mean_loss = sum(total_loss) / len(total_loss)
    with open("./result/loss_CSCNN.txt","a+") as F:
        F.writelines("{}\n".format(mean_loss))
    F.close()

def val(epoch, valloader):
    print("Epoch:{}".format(epoch))
    global best_acc
    # convert model to eval mode
    net.eval()
    correct = 0
    total = 0
    # There is no need to preserve the gradient during validation, which can save GPU memory
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            targets = targets.to(device)
            dcn_inputs = {}
            # set the model input
            for idx, key in enumerate(data_keys):
                dcn_inputs[key] = inputs[idx].to(device)

            # get forward result
            outputs = net(dcn_inputs)
            # get predict label
            _, predicted = outputs.max(1)
            total += targets.size(0)
            # Record the data for which the prediction was correct
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    with open("./result/acc_CSCNN.txt","a+") as F:
        F.writelines("{}\n".format(acc))
    F.close()

    """
    best_acc is a global variable, if the current acc is bigger than best_acc, then update best_acc, 
    otherwise skip it, so the best model can be controlled to save according to best_acc 
    """
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        # save best checkpoint
        torch.save(state, './checkpoint/DCN_best.pth')
        best_acc = acc

    print("Now acc is {}".format(acc))

if __name__ == '__main__':
    # Global variables used in training
    # best_acc: record the best validation accuracy during training
    # loss_values: record the loss during training
    best_acc=0
    loss_values = []

    print('==> Building model..')

    # Initialization DCN model
    net = DCN(use_visual=False)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net) # In case of multi-GPU environment, parallelization model is required

    # Load train dataset
    train_data = DCNDataset(json_path=r"..\CSCNN_dataset\train.json")
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    # Load validation dataset
    val_data = DCNDataset(json_path=r"..\CSCNN_dataset\val.json")
    val_dataloader = DataLoader(val_data, batch_size=4, shuffle=True)

    # Initialization loss, we use cross entropy loss
    criterion = nn.CrossEntropyLoss()
    # Initialization optimizer, we choose the learning rate=1e-5
    # Learning rate decay is used, but the effect is not obvious in Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-5, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

    # main function, train by 50 epoches
    for epoch in range(0, 50):
        train(epoch,train_dataloader)
        val(epoch,val_dataloader)