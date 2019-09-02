

import torch
from torch import nn
from torchvision import models


from models.MaskedLayer import MaskedConv2d, MaskedLinear




class AlexNetPrunable(nn.Module):
    def __init__(self, c_rate = 1.7, margin =0.1, gamma = 0.0001, num_classes=1000, power = -1,  prune_features = True, prune_classifier = True):

        super(AlexNetPrunable, self).__init__()


        self.gamma = gamma
        self.thresholds = []
        self.prune_features = prune_features
        self.prune_classifier = prune_classifier
        self.c_rate = c_rate
        self.t = margin #sarebbe half_margin
        self.power = power

        alexnet = models.alexnet(pretrained=True)

        self.relu = nn.ReLU()
        self.weight_masks = []
        self.bias_masks = []

        self.fc_masks = []
        self.fc_bias_masks = []

        self.layers = []
        #A copy of the old layers, some of the them, the ones indexed by prunable_layers_indexes, will be pruned. This structure is then used to implement the forward pass, after a trasformation
        self.pruned_layers = []
        self.prunable_layers_indexes_features = []
        self.prunable_layers_indexes_classifier = []
        self.len_features = len(alexnet.features)

        for name, param in alexnet.features.named_parameters():
            if ('weight' in name):
                self.weight_masks.append(torch.ones(param.shape).cuda())
                self.prunable_layers_indexes_features.append(int(name.split(".")[0]))

            elif ('bias' in name):
                self.bias_masks.append(torch.ones(param.shape).cuda())

        for index in range(len(alexnet.features)):
            self.layers.append(self.copy_layer(alexnet.features[index], masked = self.prune_features))
            self.pruned_layers.append(self.copy_layer(alexnet.features[index], masked=self.prune_features))


        for name_c, param_c in alexnet.classifier.named_parameters():
            if ('weight' in name_c):
                self.fc_masks.append(torch.ones(param_c.shape).cuda())
                self.prunable_layers_indexes_classifier.append(int(name_c.split(".")[0]) + self.len_features )
            elif ('bias' in name_c):
                self.fc_bias_masks.append(torch.ones(param_c.shape).cuda())


        for index_c in range(len(alexnet.classifier)):
            self.layers.append(self.copy_layer(alexnet.classifier[index_c], masked = self.prune_classifier))
            self.pruned_layers.append(self.copy_layer(alexnet.classifier[index_c], masked = self.prune_classifier))
        if prune_features:
            self.features=nn.Sequential(*self.pruned_layers[: self.len_features])
        else:
            self.features = alexnet.features
            for param in self.features.parameters():
                param.requires_grad = False

        if prune_classifier:
            self.classifier = nn.Sequential(*self.pruned_layers[self.len_features :])
        else:
            self.classifier = alexnet.classifier
            for param in self.classifier.parameters():
                param.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))


        self.prune_in_current_iteration = True
        self.count_pruning_it = 1
        self.num_classes = num_classes
        if(num_classes and num_classes != 1000):
            self.adapt_layer = nn.Linear(1000, num_classes)

        self.thresholds_features_a = []
        self.thresholds_features_b = []

        self.thresholds_classifier_a = []
        self.thresholds_classifier_b = []
        #initialize masks
        self.prune_weights()

        self.compute_thresholds()

    def copy_layer(self, layer, masked = True):
        if str(layer).split("(")[0] == 'Conv2d':
            if masked:
                new_layer = MaskedConv2d(in_channels=layer.in_channels, out_channels=layer.out_channels,
                                         kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)

            else:
                new_layer = nn.Conv2d(in_channels=layer.in_channels, out_channels = layer.out_channels, kernel_size= layer.kernel_size, stride = layer.stride, padding = layer.padding)

            new_layer.load_state_dict(layer.state_dict().copy())


        elif str(layer).split("(")[0] == 'ReLU':
            new_layer = nn.ReLU()
        elif str(layer).split("(")[0] == 'MaxPool2d':
            new_layer = nn.MaxPool2d(kernel_size=layer.kernel_size, stride=layer.stride)
        elif str(layer).split("(")[0] == 'Dropout':
            new_layer = nn.Dropout(p=layer.p)
        elif str(layer).split("(")[0] == 'Linear':
            if (masked):
                new_layer = MaskedLinear(in_features=layer.in_features, out_features=layer.out_features)
            else:
                new_layer = nn.Linear(in_features=layer.in_features, out_features=layer.out_features)

            new_layer.load_state_dict(layer.state_dict().copy())
        else:
            raise ValueError("Unkonw kind of layer: ", str(layer))

        return  new_layer.cuda()

    def forward(self, x):
        self.update_backward_weights()
        self.prepare_for_forward()
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        if(self.num_classes!=1000):
           x = self.relu(x)
           x = self.adapt_layer(x)
        self.prepare_for_backward()
        return x


    def update_backward_weights(self):
        '''
            Save backward weights that have been updated in by SGD, to use them again in backprop
            '''
        if (self.prune_features):
            for i in self.prunable_layers_indexes_features:
                self.pruned_layers[i].update_backward_weights()
        if (self.prune_classifier):
            for j in self.prunable_layers_indexes_classifier:
                self.pruned_layers[j].update_backward_weights()


    def prepare_for_forward(self):
        if (self.prune_features):
            for i in self.prunable_layers_indexes_features:
                self.pruned_layers[i].switch_mode('forward')

        if (self.prune_classifier):
            for j in self.prunable_layers_indexes_classifier:
                self.pruned_layers[j].switch_mode('forward')

    def prepare_for_backward(self):
        '''
        Switch MaskedLayers to backward mode so that the gradients are computed for all the weights and not only for the pruned ones

        '''
        if (self.prune_features):
            for i in self.prunable_layers_indexes_features:
                self.pruned_layers[i].switch_mode('backward')

        if (self.prune_classifier):
            for j in self.prunable_layers_indexes_classifier:
                self.pruned_layers[j].switch_mode('backward')


    def update_mask_and_prune_with_probability(self, iter):
        self.decide_whether_prune(iter)
        if(self.prune_in_current_iteration):
            self.update_mask()
            self.prune_weights()


    def prune_weights(self):
        #todo posso modicare direttamente self.pruned_weight, così verrano modificati anche gli elementi dentro features e classifier
        #todo attenzione ai layer di dropout nel classifier
        # preso da https: // stackoverflow.com / questions / 53544901 / how - to - mask - weights - in -pytorch - weight - parameters

        if (self.prune_features):
            index_weight = 0
            for i in self.prunable_layers_indexes_features:
                #self.pruned_layers[i].load_state_dict(self.layers[i].state_dict().copy())
                self.pruned_layers[i].set_mask(self.weight_masks[index_weight], self.bias_masks[index_weight] )

                #self.pruned_layers[i].weight.data = self.weight_masks[index_weight]* self.layers[i].weight
                #self.pruned_layers[i].bias.data = self.bias_masks[index_weight]* self.layers[i].bias
                index_weight+=1


        if (self.prune_classifier):
            index_weight = 0
            for j in self.prunable_layers_indexes_classifier:
                #self.pruned_layers[j].load_state_dict(self.layers[j].state_dict().copy())
                self.pruned_layers[j].set_mask(self.fc_masks[index_weight] , self.fc_bias_masks[index_weight])
                #self.pruned_layers[j].weight.data = self.fc_masks[index_weight]*self.layers[j].weight
                #self.pruned_layers[j].bias.data = self.fc_bias_masks[index_weight]*self.layers[j].bias
                index_weight+=1


    def compute_thresholds(self):

        if self.prune_features:
            index_mask = 0
            for i in self.prunable_layers_indexes_features:
                # Nel paper originale mean e std vengono calcolati congiuntamente per weights and bias
                # print("Dentro update mask")
                mean, std = self.compute_mean_std(self.pruned_layers[i].weight.data, self.pruned_layers[i].bias.data)

                a = (1 - self.t) * mean + self.c_rate * std
                b = (1 + self.t) * mean + self.c_rate * std

                self.thresholds_features_a.append(a)
                self.thresholds_features_b.append(b)
                index_mask+=1
            print("A_k for Features")
            print(self.thresholds_features_a)
            print("B_k for Features")
            print(self.thresholds_features_b)
        if (self.prune_classifier):
            index_mask = 0
            for j in self.prunable_layers_indexes_classifier:
                mean, std = self.compute_mean_std(self.pruned_layers[j].weight.data, self.pruned_layers[j].bias.data)
                a = (1 - self.t) * mean + self.c_rate * std
                b = (1 + self.t) * mean + self.c_rate * std

                self.thresholds_classifier_a.append(a)
                self.thresholds_classifier_b.append(b)

                index_mask += 1
            print("A_k for Classifier")
            print(self.thresholds_classifier_a)
            print("B_k for Classifer")
            print(self.thresholds_classifier_b)


    def update_mask(self):

        '''
        Basically they judge weights more than X standard deviations away from the mean as significant (denoted 'crate' in my and their code). This seems to work quite well, as layers where all weights are fairly evenly distributed not too many are cut, whilst uneven distributions will result in a larger number of weights being cut. You can see from the statistics printed to console that upper layers have more weights cut than lower layers.
        This makes sense, as upper layers are more specialised in detecting specific patterns than lower layers, which function more like Gabor filters.
        from: https://github.com/yaysummeriscoming/DNS_and_INQ

        '''


        if(self.prune_features):
            index_mask = 0
            for i in self.prunable_layers_indexes_features:
                #Nel paper originale mean e std vengono calcolati congiuntamente per weights and bias
                #print("Dentro update mask")
                under_a = (self.weight_masks[index_mask] == 1) & (
                            self.pruned_layers[i].weight.data <= self.thresholds_features_a[index_mask])
                over_b = (self.weight_masks[index_mask] == 0) & (
                        self.pruned_layers[i].weight.data > self.thresholds_features_b[index_mask])

                self.weight_masks[index_mask] = torch.where(under_a, torch.zeros(self.weight_masks[index_mask].shape).cuda(),self.weight_masks[index_mask] )
                self.weight_masks[index_mask] = torch.where(over_b, torch.ones(self.weight_masks[index_mask].shape).cuda(), self.weight_masks[index_mask])


                under_a = (self.bias_masks[index_mask] == 1) & (
                            self.pruned_layers[i].bias.data<= self.thresholds_features_a[index_mask])
                over_b  = (self.bias_masks[index_mask] == 0) & (
                        self.pruned_layers[i].bias.data> self.thresholds_features_b[index_mask])

                self.bias_masks[index_mask] = torch.where(under_a, torch.zeros(self.bias_masks[index_mask].shape).cuda(),self.bias_masks[index_mask]  )
                self.bias_masks[index_mask] = torch.where(over_b, torch.ones(self.bias_masks[index_mask].shape).cuda(), self.bias_masks[index_mask])

                index_mask+=1

        if(self.prune_classifier):
            index_mask = 0
            for j in self.prunable_layers_indexes_classifier:

                under_a = (self.fc_masks[index_mask] == 1) & (
                        self.pruned_layers[j].weight.data <= self.thresholds_classifier_a[index_mask])
                over_b = (self.fc_masks[index_mask] == 0) & (
                        self.pruned_layers[j].weight.data > self.thresholds_classifier_b[index_mask])

                self.fc_masks[index_mask] = torch.where(under_a, torch.zeros(self.fc_masks[index_mask].shape).cuda(), self.fc_masks[index_mask])
                self.fc_masks[index_mask] = torch.where(over_b, torch.ones(self.fc_masks[index_mask].shape).cuda(), self.fc_masks[index_mask])

                under_a = (self.fc_bias_masks[index_mask] == 1) & (
                        self.pruned_layers[j].bias.data <= self.thresholds_classifier_a[index_mask])
                over_b = (self.fc_bias_masks[index_mask] == 0) & (
                        self.pruned_layers[j].bias.data > self.thresholds_classifier_b[index_mask])

                self.fc_bias_masks[index_mask] = torch.where(under_a, torch.zeros(self.fc_bias_masks[index_mask].shape).cuda(), self.fc_bias_masks[index_mask])
                self.fc_bias_masks[index_mask] = torch.where(over_b, torch.ones(self.fc_bias_masks[index_mask].shape).cuda(), self.fc_bias_masks[index_mask])

                index_mask+=1

    def compute_mean_std(self, weight, bias):
        '''


        Weight and mean are jointly computed for bias and weight, as in the orginal c++ code

        '''
        n_count = torch.numel(torch.nonzero(weight)) +torch.numel(torch.nonzero(bias))
        mean = (torch.sum(torch.abs(weight)) + torch.sum(torch.abs(bias)))/n_count

        std = torch.sum(weight**2)+torch.sum(bias**2)
        std -= n_count*mean*mean
        std/=n_count
        std = torch.sqrt(std)
        return mean, std


    # todo this method should be used from the training function
    def decide_whether_prune(self, iter):

        rnd = torch.rand(1)
        prob_th = (1 + self.gamma *iter)**(-self.power)
        if (prob_th> rnd):
            self.prune_in_current_iteration =  True
        else:
            self.prune_in_current_iteration = False


    def compute_compression_rate(self):
        pruned = 0
        total = 0
        for layer in self.weight_masks:
            pruned+=torch.sum(layer)
            total+=torch.numel(layer)

        for bias in self.bias_masks:
            pruned+= torch.sum(bias)
            total+= torch.numel(bias)

        for fc in self.fc_masks:
            pruned+= torch.sum(fc)
            total+= torch.numel(fc)

        for fc_bias in self.fc_bias_masks:
            pruned+=torch.sum(fc_bias)
            total+= torch.numel(fc_bias)

        compression_rate = total/pruned
        #print("Pruned weights : {}".format(int(pruned)))
        print("Compression rate : {:.3f}".format(compression_rate.item()))

