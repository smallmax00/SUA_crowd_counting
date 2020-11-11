import torch
from torch import nn
from torch.utils import model_zoo


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vgg = VGG()
        self.load_vgg()
        self.backend = BackEnd()

        self.conv_att = BaseConv(2, 1, 1, 1, activation=None, use_bn=False)
        self.conv_att_2 = BaseConv(32, 2, 1, 1, activation=None, use_bn=False)
        self.conv_sem_central_third = BaseConv(32, 4, 1, 1, activation=nn.LeakyReLU(), use_bn=True)
        self.conv_sem_central_fourth = BaseConv(4, 1, 1, 1, activation=nn.LeakyReLU(), use_bn=True)
        self.conv_den_central_first = BaseConv(32, 8, 1, 1, activation=nn.LeakyReLU(), use_bn=True)
        self.conv_den_central_second = BaseConv(8, 1, 1, 1, activation=nn.ReLU(), use_bn=False)
        self.smooth_heaviside = Smooth_heaviside(6000, 1)

    def forward(self, input):

        input = self.vgg(input)
        backend_binary, backend_density = self.backend(*input)
        seg = self.conv_att_2(backend_binary)
        amp_out = self.conv_att(seg)

        sem_out = self.conv_sem_central_third(backend_density)
        sem_out = self.conv_sem_central_fourth(sem_out)

        dmp_out_middle = backend_density * amp_out
        dmp_out = self.conv_den_central_first(dmp_out_middle)
        dmp_out = dmp_out * sem_out
        dmp_out = self.conv_den_central_second(dmp_out)
        dmp_to_att = self.smooth_heaviside(dmp_out)
        return dmp_out, dmp_to_att, seg

    def load_vgg(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
        new_dict = {}
        for i in range(13):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']

        self.vgg.load_state_dict(new_dict)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        conv3_1 = self.conv3_1(input)
        input = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(input)

        input = self.pool(conv3_3)
        conv4_1 = self.conv4_1(input)
        input = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(input)

        input = self.pool(conv4_3)
        conv5_1 = self.conv5_1(input)
        input = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(input)

        return conv2_2, conv3_1, conv3_3, conv4_1, conv4_3, conv5_1, conv5_3


class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = BaseConv(1024, 256, 1, 1, activation=nn.LeakyReLU(), use_bn=True)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.LeakyReLU(), use_bn=True)

        self.conv3 = BaseConv(512, 128, 1, 1, activation=nn.LeakyReLU(), use_bn=True)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.LeakyReLU(), use_bn=True)

        self.conv5 = BaseConv(256, 64, 1, 1, activation=nn.LeakyReLU(), use_bn=True)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.LeakyReLU(), use_bn=True)
        self.conv7 = BaseConv(64, 32, 3, 1, activation=None, use_bn=True)

        self.att_1_first = self.att_layer([512, 512, 512])
        self.att_2_first = self.att_layer([512, 512, 512])
        self.att_3_first = self.att_layer([256, 256, 256])

        self.att_1 = self.att_layer([1024, 512, 512])
        self.att_2 = self.att_layer([1024, 512, 512])
        self.att_3 = self.att_layer([512, 256, 256])
        self.dropout = nn.Dropout(p=0.5)

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def forward(self, *input):
        conv2_2, conv3_1, conv3_3, conv4_1, conv4_3, conv5_1, conv5_3 = input

        input_t1 = self.upsample(conv5_3)
        input_t1 = torch.cat([input_t1, conv4_3], 1)
        input_t1 = self.conv1(input_t1)
        input_t1 = self.conv2(input_t1)
        input_t1 = self.upsample(input_t1)
        input_t1 = torch.cat([input_t1, conv3_3], 1)
        input_t1 = self.conv3(input_t1)
        input_t1 = self.conv4(input_t1)
        input_t1 = self.upsample(input_t1)
        input_t1 = torch.cat([input_t1, conv2_2], 1)
        input_t1 = self.conv5(input_t1)
        input_t1 = self.conv6(input_t1)
        input_t1 = self.conv7(input_t1)
        input_t1 = self.upsample(input_t1)

        input_t1 = self.dropout(input_t1)

        input_t3 = self.upsample(conv5_3)
        input_t3 = torch.cat([input_t3, conv4_3], 1)
        input_t3 = self.conv1(input_t3)
        input_t3 = self.conv2(input_t3)
        input_t3 = self.upsample(input_t3)
        input_t3 = torch.cat([input_t3, conv3_3], 1)
        input_t3 = self.conv3(input_t3)
        input_t3 = self.conv4(input_t3)
        input_t3 = self.upsample(input_t3)
        input_t3 = torch.cat([input_t3, conv2_2], 1)
        input_t3 = self.conv5(input_t3)
        input_t3 = self.conv6(input_t3)
        input_t3 = self.conv7(input_t3)
        input_t3 = self.upsample(input_t3)

        input_t3 = self.dropout(input_t3)


        return input_t1, input_t3


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


class Smooth_heaviside(nn.Module):
    def __init__(self, k=None, m=None):
        super(Smooth_heaviside, self).__init__()
        self.k = k
        self.m = m
    def forward(self, x):
        x1 = 2 - 1 / (torch.sigmoid(self.k * x) / self.m)
        x2 = torch.sigmoid(self.k * x)
        return x1 * x2
