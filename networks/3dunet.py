import torch
import torch.nn as nn
import torch.optim
from utils.run_util import print_size

class BasicUNet(nn.Module):
    """
    Basic 3D UNet
    """ 
    def __init__(self, in_channels, num_classes, base_n_filter = 32): 
        super(BasicUNet, self).__init__()
        
        self.encoder0 = self.encoder(in_channels, base_n_filter*2)
        self.encoder1 = self.encoder(base_n_filter*2, base_n_filter*4)
        self.encoder2 = self.encoder(base_n_filter*4, base_n_filter*8)

        self.bneck0 = self.bottleneck(base_n_filter*8, base_n_filter*8)
        self.bneck1 = self.bottleneck(base_n_filter*8, base_n_filter*16)

        self.decoder0 = self.decoder(base_n_filter*24, base_n_filter*8, ksize=4)
        self.decoder1 = self.decoder(base_n_filter*12, base_n_filter*4, ksize=4, out_pad=1)
        self.decoder2 = self.decoder(base_n_filter*6, base_n_filter*2, ksize=4, out_pad=1)

        self.output_ = self.output(base_n_filter*2, num_classes)

    def encoder(self, in_channels, out_channels): 
        mid_layer_filter = int(out_channels/2)
        return nn.Sequential(
            nn.Conv3d(in_channels = in_channels, out_channels = mid_layer_filter, kernel_size = 4, stride=1),
            nn.ReLU(0.3),
            nn.Conv3d(in_channels =  mid_layer_filter, out_channels = out_channels, kernel_size = 4, stride=2, padding = 1 ),
            nn.BatchNorm3d(num_features = out_channels), 
            nn.ReLU(0.3)
        )

    def bottleneck(self, in_channels, out_channels): 
        return nn.Sequential(
            nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 4, stride = 1, padding = 'same'),
            nn.InstanceNorm3d(num_features = out_channels), 
            nn.LeakyReLU(0.3), 
            nn.Dropout3d(p = 0.2)
        )

    def decoder(self, in_channels, out_channels, ksize, out_pad =0): 
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels = in_channels, out_channels = out_channels, kernel_size = ksize, stride = 2, padding=1, output_padding=out_pad), 
            nn.ReLU(0.3),
            nn.ConvTranspose3d(in_channels = out_channels, out_channels = out_channels, kernel_size = ksize, stride = 1), 
            nn.BatchNorm3d(num_features = out_channels),
            nn.ReLU(0.3)
        )
    
    def output(self,in_channels, out_channels): 
        return nn.Sequential(
            nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 1), 
        )

    def forward(self, x): 
        """
        Model execution 
        """
        down1 = self.encoder0(x)
        down2 = self.encoder1(down1)
        down3 = self.encoder2(down2)
       
        bot = self.bneck0(down3)
        bot = self.bneck1(bot)
        
        up1 = torch.cat([bot, down3], dim=1)
        up1 = self.decoder0(up1)	 
        
        up2 = torch.cat([up1, down2], dim=1)
        up2 = self.decoder1(up2)
        up3 = torch.cat([up2, down1], dim =1)
        up3 = self.decoder2(up3)

        out = self.output_(up3)
        return out
