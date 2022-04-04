import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from model.network import DCN
import torch
from dataset.DCN_dataset import DCNDataset
from torch.utils.data import Dataset, DataLoader

def test(epoch, testloader):
    print("Epoch:{}".format(epoch))
    global best_acc
    net.eval()
    correct = 0
    total = 0
    labels = []
    predict_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets = targets.to("cpu")
            dcn_inputs = {}
            labels.append(targets.item())

            dcn_inputs["blink"] = inputs[0]
            dcn_inputs["yawn"] = inputs[1]
            dcn_inputs["gaze_x"] = inputs[2]
            dcn_inputs["gaze_y"] = inputs[3]
            dcn_inputs["heart_rate"] = inputs[4]
            dcn_inputs["images"] = inputs[5]

            outputs = net(dcn_inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            predict_labels.append(predicted.item())
    acc = 100. * correct / total

    print("Now acc is {}".format(acc))
    return labels,predict_labels


net = DCN(use_visual=False)
net = net.to("cpu")

checkpoint = torch.load('./checkpoint/DCN_best.pth')
net.load_state_dict(checkpoint["net"])

test_data = DCNDataset(json_path=r"D:\Chrome Download\CSCNN_dataset\test.json")
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

for epoch in range(0, 1):
    labels,predict_labels = test(epoch, test_dataloader)

labels = np.array(labels[:]).reshape(-1,)
predict_labels = np.array(predict_labels[:]).reshape(-1,)
from sklearn.metrics import precision_score, recall_score,roc_auc_score
print(precision_score(labels,predict_labels,average='macro'))
print(recall_score(labels,predict_labels,average='macro'))
# print(roc_auc_score(labels,predict_labels,average='macro',multi_class="ovr"))
classes = ["1", "2", "3"]
confusion_mat = confusion_matrix(labels, predict_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
disp.plot(
    include_values=True,
    cmap="viridis",
    ax=None,
    xticks_rotation="horizontal",
    values_format="d"
)
plt.show()