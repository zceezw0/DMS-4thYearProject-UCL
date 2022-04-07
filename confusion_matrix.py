import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score

from model.network import DCN
from dataset.DCN_dataset import DCNDataset

# init data keys, use when training and validation.
data_keys = ["blink","yawn","gaze_x","gaze_y","heart_rate","images"]
# get current device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# our label class
classes = ["1", "2", "3"]

def test(testloader):
    net.eval()
    correct = 0
    total = 0
    labels = []
    predict_labels = []
    # There is no need to preserve the gradient during validation, which can save GPU memory
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets = targets.to("cpu")
            dcn_inputs = {}
            labels.append(targets.item())

            for idx, key in enumerate(data_keys):
                dcn_inputs[key] = inputs[idx].to(device)

            # get forward result
            outputs = net(dcn_inputs)
            # get predict label
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            predict_labels.append(predicted.item())
    acc = 100. * correct / total

    print("Now acc is {}".format(acc))
    return labels,predict_labels

if __name__ == '__main__':
    # init model and load checkpoint
    net = DCN(use_visual=False)
    net = net.to(device)
    checkpoint = torch.load('./checkpoint/DCN_best.pth')
    net.load_state_dict(checkpoint["net"])

    # Load test dataset
    test_data = DCNDataset(json_path=r"D:\Chrome Download\CSCNN_dataset\test.json")
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    # get ground truth and predict label
    labels, predict_labels = test(test_dataloader)
    labels = np.array(labels[:]).reshape(-1, )
    predict_labels = np.array(predict_labels[:]).reshape(-1, )

    # caculate precision score and recall score
    print(precision_score(labels, predict_labels, average='macro'))
    print(recall_score(labels, predict_labels, average='macro'))

    # plot confusion_matrix according to the ground truth and predict label
    confusion_mat = confusion_matrix(labels, predict_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
    disp.plot(include_values=True,
              cmap="viridis",
              ax=None,
              xticks_rotation="horizontal",
              values_format="d")
    plt.show()