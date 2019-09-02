import torch
from torch import nn

import torch.nn.functional as F

from models.MaskedLayer import MaskedConv2d, MaskedLinear
from models.LeNet import LeNet


class LeNetPrunable(nn.Module):
    '''
    Rispetto ad AlexNetPrunable, non è necessario prunare seperatamente classifier e features
    '''

    def __init__(self, pretrained_model_path, c_rate = 4, margin =0.1, gamma = 0.0001, power = -1 ):
        super(LeNetPrunable, self).__init__()
        self.gamma = gamma
        self.thresholds = []
        self.c_rate = c_rate
        self.c_rate_conv = c_rate
        self.t = margin  # sarebbe half_margin
        self.power = power
        LeNet = LeNet()
        state_dict = torch.load(pretrained_model_path)
        LeNet.load_state_dict(state_dict)

        #self.relu = nn.ReLU()
        self.weight_masks = []
        self.bias_masks = []

        #self.fc_masks = []
        #self.fc_bias_masks = []
        self.pruned_layers =[]

        for name, param in LeNet.named_parameters():
            if ('weight' in name):
                self.weight_masks.append(torch.ones(param.shape).cuda())
                #self.prunable_layers_indexes_features.append(int(name.split(".")[0]))

            elif ('bias' in name):
                self.bias_masks.append(torch.ones(param.shape).cuda())

        self.conv1 = self.copy_layer(LeNet.conv1)
        self.pruned_layers.append(self.conv1)

        self.conv2 = self.copy_layer(LeNet.conv2)
        self.pruned_layers.append(self.conv2)

        self.fc1 = self.copy_layer(LeNet.fc1)
        self.pruned_layers.append(self.fc1)

        self.fc2 = self.copy_layer(LeNet.fc2)
        self.pruned_layers.append(self.fc2)

        self.prune_in_current_iteration = True
        self.pruning_prob = 1
        self.thresholds_a = []
        self.thresholds_b = []
        # initialize masks
        self.prune_weights()

        self.compute_thresholds()

        self.freeze_prune = False
    def copy_layer(self, layer):
        if str(layer).split("(")[0] == 'Conv2d':

            new_layer = MaskedConv2d(in_channels=layer.in_channels, out_channels=layer.out_channels,
                                         kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
            new_layer.load_state_dict(layer.state_dict().copy())

        elif str(layer).split("(")[0] == 'ReLU':
            new_layer = nn.ReLU()
        elif str(layer).split("(")[0] == 'MaxPool2d':
            new_layer = nn.MaxPool2d(kernel_size=layer.kernel_size, stride=layer.stride)
        elif str(layer).split("(")[0] == 'Dropout':
            new_layer = nn.Dropout(p=layer.p)
        elif str(layer).split("(")[0] == 'Linear':

            new_layer = MaskedLinear(in_features=layer.in_features, out_features=layer.out_features)
            new_layer.load_state_dict(layer.state_dict().copy())
        else:
            raise ValueError("Unkonwn kind of layer: ", str(layer))

        return  new_layer.cuda()


    def forward(self, x):

        self.update_backward_weights()
        self.prepare_for_forward()

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        self.prepare_for_backward()
        return x
        '''
        if self.distillation:
            return x
        else:
            return F.log_softmax(x, dim=1)
        '''

    def prepare_for_backward(self):
        for layer in self.pruned_layers:
            layer.switch_mode('backward')

    def prepare_for_forward(self):
        for layer in self.pruned_layers:
            layer.switch_mode('forward')

    def update_backward_weights(self):
        '''
        Save backward weights that have been updated in by SGD, to use them again in backprop
        '''
        for layer in self.pruned_layers:
            layer.update_backward_weights()

    def prune_weights(self):
        for index in range(len(self.pruned_layers)):
            self.pruned_layers[index].set_mask(self.weight_masks[index], self.bias_masks[index])

    def update_mask(self):

        for index in range(len(self.pruned_layers)):
            under_a = (self.weight_masks[index] == 1 ) & (self.pruned_layers[index].weight.data <= self.thresholds_a[index])
            over_b = (self.weight_masks[index] == 0) & (self.pruned_layers[index].weight.data > self.thresholds_b[index])

            self.weight_masks[index] = torch.where(under_a, torch.zeros(self.weight_masks[index].shape).cuda(), self.weight_masks[index])
            self.weight_masks[index] = torch.where(over_b, torch.ones(self.weight_masks[index].shape).cuda(), self.weight_masks[index])



            under_a = (self.bias_masks[index] == 1) & (
                    self.pruned_layers[index].bias.data <= self.thresholds_a[index])
            over_b = (self.bias_masks[index] == 0) & (
                    self.pruned_layers[index].bias.data > self.thresholds_b[index])

            self.bias_masks[index] = torch.where(under_a,
                                                      torch.zeros(self.bias_masks[index].shape).cuda(),
                                                      self.bias_masks[index])
            self.bias_masks[index] = torch.where(over_b,
                                                      torch.ones(self.bias_masks[index].shape).cuda(),
                                                   self.bias_masks[index])


    def compute_thresholds(self):
        for index in range(len(self.pruned_layers)):
            mean, std = self.compute_mean_std(self.pruned_layers[index].weight.data, self.pruned_layers[index].bias.data)
            if (index < 2):
                a = (1 - self.t) * mean + self.c_rate_conv * std
                b = (1 + self.t) * mean + self.c_rate_conv * std
            else:
                a = (1 - self.t) * mean + self.c_rate * std
                b = (1 + self.t) * mean + self.c_rate * std
            self.thresholds_a.append(a)
            self.thresholds_b.append(b)
        print("Thresholds")
        print(self.thresholds_a)
        print(self.thresholds_b)


    def compute_mean_std(self, weight, bias):
        '''
        Weight and mean are jointly computed for bias and weight, as in the orginal c++ code
        '''
        n_count = torch.numel(torch.nonzero(weight)) +torch.numel(torch.nonzero(bias))
        #mean = (torch.sum(torch.abs(weight)) + torch.sum(torch.abs(bias)))/n_count
        mean = (torch.sum(torch.abs(weight)) )/n_count

        std = torch.sum(weight**2) +torch.sum(bias**2)
        std -= n_count*mean*mean
        std/=n_count
        std = torch.sqrt(std)
        return mean, std.item()


    def compute_compression_rate(self):
        pruned = 0
        total = 0
        for layer in self.weight_masks:
            pruned+=torch.sum(layer)
            total+=torch.numel(layer)

        for bias in self.bias_masks:
            pruned+= torch.sum(bias)
            total+= torch.numel(bias)


        compression_rate = total/pruned
        #print("Pruned weights : {}".format(int(pruned)))
        #print("Compression rate : {:.3f}".format(compression_rate.item()))
        return compression_rate.item()


    def update_mask_and_prune_with_probability(self, iter):

        if(self.decide_whether_prune(iter)):
            self.update_mask()
            self.prune_weights()

    def decide_whether_prune(self, iter):

        rnd = torch.rand(1).item()
        self.pruning_prob = (1 + self.gamma *iter)**(-self.power)
        '''
        if (self.pruning_prob < 0.2):
            self.freeze_prune = True
            return False
        '''
        if (self.pruning_prob> rnd):
            self.prune_in_current_iteration = True

            return True
        else:
            self.prune_in_current_iteration = False
            return False



