import torch
import torch.nn as nn

class DDCB(nn.Module):
    def __init__(self, in_planes):
        super(DDCB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, 256, 1), 
            nn.ReLU(True), 
            nn.Conv2d(256, 64, 3, padding = 1), 
            nn.ReLU(True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes + 64, 256, 1), 
            nn.ReLU(True), 
            nn.Conv2d(256, 64, 3, padding = 2, dilation = 2), 
            nn.ReLU(True)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_planes + 128, 256, 1), 
            nn.ReLU(True), 
            nn.Conv2d(256, 64, 3, padding = 3, dilation = 3), 
            nn.ReLU(True)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_planes + 128, 512, 3, padding = 1), 
            nn.ReLU(True)
            )
    
    def forward(self, x):
        x1_raw = self.conv1(x)
        x1 = torch.cat([x, x1_raw], 1)
        x2_raw = self.conv2(x1)
        x2 = torch.cat([x, x1_raw, x2_raw], 1)
        x3_raw = self.conv3(x2)
        x3 = torch.cat([x, x2_raw, x3_raw], 1)
        output = self.conv4(x3)
        return output


class DenseScaleNet(nn.Module):
    def __init__(self, load_model='', pretrained_model=''):
        super(DenseScaleNet, self).__init__()
        self.load_model = load_model
        self.pretrained_model = pretrained_model
        self.features_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,]
        self.features = make_layers(self.features_cfg)
        self.DDCB1 = DDCB(512)
        self.DDCB2 = DDCB(512)
        self.DDCB3 = DDCB(512)

        self.dense_output_layers = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1), 
            nn.ReLU(True), 
            nn.Conv2d(128, 64, 3, padding=1), 
            nn.ReLU(True), 
            nn.Conv2d(64, 1, 1)
            )

        self.depth_output_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(num_features=1024), 
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
            )

        self._initialize_weights()

    def sharelayer(self, x):
        x = self.features(x)
        x1_raw = self.DDCB1(x)
        x1 = x1_raw + x
        x2_raw = self.DDCB2(x1)
        x2 = x2_raw + x1_raw + x
        x3_raw = self.DDCB3(x2)
        x3 = x3_raw + x2_raw + x1_raw + x
        return x3

    def forward(self, x1=None, x2=None, phase="train"):
        if not self.load_model and phase == "train":
            h1_shared = self.sharelayer(x1)
            h2_shared = self.sharelayer(x2)
            output1 = self.dense_output_layers(h1_shared)
            output2 = self.depth_output_layers(h2_shared)
            return output1, output2
        else:
            h1_shared = self.sharelayer(x1)
            output1 = self.dense_output_layers(h1_shared)
            return output1

    def _initialize_weights(self):
        self_dict = self.state_dict()
        pretrained_dict = dict()
        self._random_initialize_weights()
        
        if not self.load_model and self.pretrained_model:
            vgg16 = torch.load(self.pretrained_model)
            for k, v in vgg16.items():
                if k in self_dict and self_dict[k].size() == v.size():
                    pretrained_dict[k] = v
            self_dict.update(pretrained_dict)
            self.load_state_dict(self_dict)
        else:
            self.load_state_dict(torch.load(self.load_model))

    def _random_initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                #nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)