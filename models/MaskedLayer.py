import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)



class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)

        self.mask_flag = False


    def set_mask(self, mask, bias_mask):
        self.mask = to_var(mask, requires_grad=False)
        self.bias_mask = to_var(bias_mask, requires_grad=False)
        #self.weight.data = self.backward_weight * self.mask
        #self.bias.data = self.backward_bias*self.bias_mask
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def switch_mode(self, mode):
        if mode == "backward":
            self.weight.data = self.backward_weight
            self.bias.data = self.backward_bias
        elif mode == "forward":
            self.weight.data = self.backward_weight * self.mask.data
            self.bias.data = self.backward_bias * self.bias_mask.data


    def forward(self, x):

        '''
        if self.mask_flag == True:
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)
        '''
        return F.linear(x, self.weight, self.bias)



    def load_state_dict(self, state_dict, strict=True):
        super(MaskedLinear, self).load_state_dict(state_dict, strict=strict)
        self.backward_weight = self.weight.data.clone()
        self.backward_bias = self.bias.data.clone()
        if torch.cuda.is_available():

            self.backward_weight = self.backward_weight.cuda()
            self.backward_bias = self.backward_weight.cuda()

    #Salva i pesi aggiornati nell'iteazione corrente, di modo da usarli nella successiva
    def update_backward_weights(self):
        self.backward_weight = self.weight.data.clone()
        self.backward_bias = self.bias.data.clone()
        if torch.cuda.is_available():
            self.backward_weight = self.backward_weight.cuda()
            self.backward_bias =  self.backward_bias.cuda()


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride, padding, dilation, groups, bias)

        self.mask_flag = False

    def set_mask(self, mask, bias_mask):
        self.mask = to_var(mask, requires_grad=False)
        self.bias_mask = to_var(bias_mask, requires_grad=False)

        #self.weight.data = self.backward_weight * self.mask.data
        #self.bias.data = self.backward_bias * self.bias_mask
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def switch_mode(self, mode):
        if mode == "backward":
            self.weight.data = self.backward_weight
            self.bias.data = self.backward_bias

        elif mode == "forward":
            self.weight.data = self.backward_weight * self.mask.data
            self.bias.data = self.backward_bias * self.bias_mask.data

    def update_backward_weights(self):
        self.backward_weight = self.weight.data.clone().to('cuda:0')
        self.backward_bias = self.bias.data.clone().to('cuda:0')

    def forward(self, x):

        '''

        if self.mask_flag == True:
            #weight = self.weight * self.mask
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        '''
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def load_state_dict(self, state_dict, strict=True):
        super(MaskedConv2d, self).load_state_dict(state_dict,strict=strict)
        self.backward_weight = self.weight.data.clone().to('cuda:0')
        self.backward_bias = self.bias.data.clone().to('cuda:0')

