import torch
import torch.nn as nn
import torch.optim

class Discriminator(nn.Module): 
    def __init__(self, in_channels, num_classes=1): 
        super().__init__()
        self.encoder0 = self.encoder(in_channels*2, 64)
        self.encoder1 = self.encoder(64, 128)
        self.encoder2 = self.encoder(128, 256)
        self.encoder3 = self.encoder(256, 512)
        self.output_ = self.output(512, num_classes)

    def encoder(self, in_channels, out_channels): 
        return nn.Sequential(
            nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 4, stride=2, padding = 1 ),
            nn.InstanceNorm3d(num_features= out_channels),
            nn.LeakyReLU()
        )

    def output(self, in_channels, out_channels): 
        return nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size = 4, stride = 1, padding = 'same')
        )

    def print_size(self, name, x): 
        print('{} size: {}'.format(name, x.size()))

    def forward(self, brain, mask): 
        out = torch.cat([brain, mask], dim=1)
        # self.print_size('output 0', out)
        out = self.encoder0(out)
        # self.print_size('output 1', out)
        out = self.encoder1(out)
        # self.print_size('output 2', out)
        out = self.encoder2(out)
        # self.print_size('output 3', out)
        out = self.encoder3(out)
        # self.print_size('output 4', out)
        out = self.output_(out)
        # self.print_size('output 5', out)
        return out

        
