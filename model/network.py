import torch
from torchvision import models
import torch.nn as nn
import numpy as np

class DeepNet(nn.Module):
    """
    DeepNet is a multi-layer perceptron with added BN,
    core idea is:
            y_{i+1} = Dropout(Relu(BN(Linear(y_i))))
    """
    def __init__(self, input_size, init_std=0.2, dropout_rate=0.1):
        super(DeepNet, self).__init__()
        self.init_std = init_std

        #Deep Layer I
        self.linear1 = nn.Linear(input_size, input_size)
        self.bn1 = nn.BatchNorm1d(input_size)
        self.activation1 = nn.ReLU()
        self.droput1 = nn.Dropout(dropout_rate)

        # Deep Layer II
        self.linear2 = nn.Linear(input_size, input_size // 2)
        self.bn2 = nn.BatchNorm1d(input_size // 2)
        self.activation2 = nn.ReLU()
        self.droput2 = nn.Dropout(dropout_rate)

        # Deep Layer III
        self.linear3 = nn.Linear(input_size // 2, input_size // 2)
        self.bn3 = nn.BatchNorm1d(input_size // 2)
        self.activation3 = nn.ReLU()
        self.droput3 = nn.Dropout(dropout_rate)

        # Deep Layer IV
        self.linear4 = nn.Linear(input_size // 2, input_size // 4)

        self.init_weight()

    def init_weight(self):
        # init weight in Linear and BatchNorm Layers
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0,self.init_std)
                layer.bias.data.zero_()
            if isinstance(layer,nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def deep_layer(self, input, linear, bn, activation):
        # One layer in MLP
        intermediate = linear(input)
        intermediate = bn(intermediate)
        intermediate = activation(intermediate)
        return intermediate

    def forward(self, input):
        #Deep Layer I
        intermediate = self.deep_layer(input, self.linear1, self.bn1, self.activation1)
        intermediate = self.droput1(intermediate)

        # Deep Layer II
        intermediate = self.deep_layer(intermediate, self.linear2, self.bn2, self.activation2)
        intermediate = self.droput2(intermediate)

        # Deep Layer III
        intermediate = self.deep_layer(intermediate, self.linear3, self.bn3, self.activation3)
        intermediate = self.droput3(intermediate)

        return self.linear4(intermediate)

class CrossNet(nn.Module):
    """
    The core idea of CrossNet:
            y_(i+1) = weight * y_i + b + y_i
    i represents the current layer,
    w represents the weight,
    b represents the bias,
    y represents the tensor of the current layer
    """
    def __init__(self, input_size, init_std=1e-4):
        super(CrossNet, self).__init__()
        self.init_std = init_std
        # Cross Layer I
        self.weight1 = nn.Parameter(torch.Tensor(input_size,1))
        self.bias1 = nn.Parameter(torch.Tensor(input_size,1))
        # Cross Layer II
        self.weight2 = nn.Parameter(torch.Tensor(input_size, 1))
        self.bias2 = nn.Parameter(torch.Tensor(input_size, 1))
        # Cross Layer III
        self.weight3 = nn.Parameter(torch.Tensor(input_size, 1))
        self.bias3 = nn.Parameter(torch.Tensor(input_size, 1))

        self.init_weight()

    def init_weight(self):
        # init weight in Weight and Bias
        init_weight_layers = [self.weight1,self.bias1,
                              self.weight2,self.bias2,
                              self.weight3,self.bias3]
        for layer in init_weight_layers:
            nn.init.normal_(layer,mean=0,std=self.init_std)

    def cross_layer(self, input, weight, bias):
        batch_size = input.size()[0]
        emb_size = input.size()[1]
        input_T = input.view(batch_size, 1, emb_size)
        # core operation
        cross_layer_output = (input.matmul(input_T)).matmul(weight) + bias + input
        return cross_layer_output

    def forward(self, cross_input):
        input = cross_input.unsqueeze(-1)
        cross_output1 = self.cross_layer(input, self.weight1, self.bias1)
        cross_output2 = self.cross_layer(cross_output1, self.weight2, self.bias2)
        cross_output3 = self.cross_layer(cross_output2, self.weight3, self.bias3)
        return cross_output3.view(cross_output3.size()[0], -1)


class ChannelAttention(nn.Module):
    def __init__(self, channel, category_blink, category_yawn, reduction=4):
        super().__init__()
        self.channel = channel
        # Two pooling layers to obtain information of different meanings
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # The embedding layer of the two features "blink" and "yawn"
        self.blink_emb_dict = nn.Embedding(category_blink, channel//2)
        self.yawn_emb_dict = nn.Embedding(category_yawn, channel//2)

        # Squeeze and excitation
        self.se = nn.Sequential(
            nn.Conv2d(channel * 2, channel * 2 // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel * 2 // reduction, channel, 1, bias=False)
        )
        # Normalize the weights by sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, blink_input, yawn_input):
        # Get "blink" and "yawn" embedding
        blink_emb = self.blink_emb_dict(blink_input).view(input.size()[0],self.channel//2,1,1)
        yawn_emb = self.yawn_emb_dict(yawn_input).view(input.size()[0],self.channel//2,1,1)

        # Get the pooling result in the channel dimension
        # Concat pooling result, "blink" embedding, "yawn" embedding
        avg_input = torch.cat([self.avgpool(input),blink_emb,yawn_emb],dim=1)
        max_input = torch.cat([self.maxpool(input), blink_emb, yawn_emb], dim=1)
        # Squeeze and excitation
        # 1x1x(C1+C2+C3) to 1x1xC1
        avg_output = self.se(avg_input)
        max_output = self.se(max_input)
        # The weights are normalized and multiplied back to the original input for attention in the channel dimension
        output = self.sigmoid(avg_output + max_output) * input
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        # Normalize the weights by sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # similar operator like channel attention
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output) * x
        return output

class ResNetCSCNN(nn.Module):
    def __init__(self, category_blink, category_yawn, visual_size, cscnn_inchannel=64):
        super(ResNetCSCNN, self).__init__()
        self.visual_size = visual_size
        # After the first convolutional layer and pooling layer of resnet18, access a CSCNN layer
        self.resnet18 = models.resnet18(pretrained=False)
        self.input_layer = nn.Sequential(*list(self.resnet18.children()))[:4]
        self.cscnn = ChannelAttention(cscnn_inchannel, category_blink, category_yawn)
        self.output_layer = nn.Sequential(*list(self.resnet18.children()))[4:-1]

        # Reduce the final output to the visual size dimension
        self.fc = nn.Linear(512, self.visual_size)

    def forward(self, input, blink_input, yawn_input):
        # Get visual embedding
        batch_size = input.size()[0]
        intermediate = self.input_layer(input)
        cscnn_output = self.cscnn(intermediate, blink_input, yawn_input)
        output = self.output_layer(cscnn_output).view(batch_size,-1)
        output = self.fc(output)
        return output

class DCN(nn.Module):
    def __init__(self, use_visual=True, init_std=1e-4, visual_size=128, embedding_size=8):
        super(DCN, self).__init__()
        self.use_visual = use_visual
        self.init_std = init_std
        # number of categories of features
        categorys = [5,2,5,5,5]

        # visual encoder for get visual embedding
        self.visual_encoder = ResNetCSCNN(5, 2, visual_size)

        # init embedding layer for sparse feature
        self.blink_emb = nn.Embedding(categorys[0], embedding_size)
        self.yawn_emb = nn.Embedding(categorys[1], embedding_size)
        self.gazex_emb = nn.Embedding(categorys[2], embedding_size)
        self.gazey_emb = nn.Embedding(categorys[3], embedding_size)
        self.heart_emb = nn.Embedding(categorys[4], embedding_size)
        self.init_weight()

        # get deepnet input size and crossnet input size
        deep_size = len(categorys) * embedding_size
        cross_size = len(categorys) * embedding_size
        if self.use_visual:
            # For ablation study, we can also do a comparison without using visual embedding
            deep_size += visual_size

        # init DCN model
        self.crossnet = CrossNet(cross_size)
        self.deepnet = DeepNet(deep_size)

        # output layer
        self.classifier = nn.Linear(deep_size//4+cross_size,3)

    def init_weight(self):
        # init weight for embedding layers
        emb_dicts = [self.blink_emb,self.yawn_emb,self.gazex_emb,self.gazey_emb,self.heart_emb]
        for layer in emb_dicts:
            layer.weight.data.normal_(0, self.init_std)

    def embedding_dict(self, data):
        # forward embedding layers
        blink_input = data["blink"]
        yawn_input = data["yawn"]
        gaze_x = data["gaze_x"]
        gaze_y = data["gaze_y"]
        heart_rate = data["heart_rate"]

        blink_emb = self.blink_emb(blink_input)
        yawn_emb = self.yawn_emb(yawn_input)
        gazex_emb = self.gazex_emb(gaze_x)
        gazey_emb = self.gazey_emb(gaze_y)
        heart_emb = self.heart_emb(heart_rate)

        return torch.cat([blink_emb,yawn_emb,gazex_emb,gazey_emb,heart_emb],dim=1)

    def forward(self, data):
        images = data["images"]
        # get sparse embeddings
        sparse_feature = self.embedding_dict(data)

        # get dense features
        # if use visual embedding: dense feature = visual embedding + sparse feature embeddings
        # if not use visual embedding: dense feature = sparse feature embeddings
        dense_feature = sparse_feature
        if self.use_visual:
            visual_emb = self.visual_encoder(images, data["blink"], data["yawn"])
            dense_feature = torch.cat([dense_feature,visual_emb],dim=1)

        # forward DCN model
        cross_output = self.crossnet(sparse_feature)
        deep_output = self.deepnet(dense_feature)

        # output classifier result to calculate loss or predict label
        output = torch.cat([cross_output,deep_output],dim=1)
        output = self.classifier(output)
        return output


if __name__ == '__main__':
    # Test
    model = DCN()
    data = {}
    data["images"] = torch.randn((2, 3, 224, 224))
    data["blink"] = torch.tensor(np.array([1, 2]), dtype=torch.long)
    data["yawn"] = torch.tensor(np.array([1, 2]), dtype=torch.long)
    data["gaze_x"] = torch.tensor(np.array([1, 2]), dtype=torch.long)
    data["gaze_y"] = torch.tensor(np.array([1, 2]), dtype=torch.long)
    data["heart_rate"] = torch.tensor(np.array([1, 2]), dtype=torch.long)
    output = model(data)