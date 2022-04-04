import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import os
from model.network import DCN
from dataset.DCN_dataset import DCNDataset
from torch.utils.data import Dataset, DataLoader

def train(epoch, trainloader):
    global loss_values
    print("Epoch:{}".format(epoch))
    net.train()
    total_loss = []
    for data in tqdm(trainloader, leave=False, total=len(trainloader)):
        inputs=data[0]
        targets=data[1]
        dcn_inputs = {}

        dcn_inputs["blink"] = inputs[0]
        dcn_inputs["yawn"] = inputs[1]
        dcn_inputs["gaze_x"] = inputs[2]
        dcn_inputs["gaze_y"] = inputs[3]
        dcn_inputs["heart_rate"] = inputs[4]
        dcn_inputs["images"] = inputs[5]

        targets=targets.to(device)
        optimizer.zero_grad()

        outputs = net(dcn_inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        loss_value = loss.item()
        loss_values.append(loss_value)
        total_loss.append(loss_value)

        optimizer.step()
    mean_loss = sum(total_loss) / len(total_loss)
    with open("./result/loss_CSCNN.txt","a+") as F:
        F.writelines("{}\n".format(mean_loss))
    F.close()

def test(epoch, testloader):
    print("Epoch:{}".format(epoch))
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets = targets.to(device)
            dcn_inputs = {}

            dcn_inputs["blink"] = inputs[0]
            dcn_inputs["yawn"] = inputs[1]
            dcn_inputs["gaze_x"] = inputs[2]
            dcn_inputs["gaze_y"] = inputs[3]
            dcn_inputs["heart_rate"] = inputs[4]
            dcn_inputs["images"] = inputs[5]

            outputs = net(dcn_inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    with open("./result/acc_CSCNN.txt","a+") as F:
        F.writelines("{}\n".format(acc))
    F.close()

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/DCN_best.pth')
        best_acc = acc

    print("Now acc is {}".format(acc))

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc=0
    loss_values = []

    print('==> Building model..')
    net = DCN(use_visual=False)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    train_data = DCNDataset(json_path=r"D:\Chrome Download\CSCNN_dataset\train.json")
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)

    test_data = DCNDataset(json_path=r"D:\Chrome Download\CSCNN_dataset\test.json")
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-5, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

    for epoch in range(0, 50):
        train(epoch,train_dataloader)
        test(epoch,test_dataloader)