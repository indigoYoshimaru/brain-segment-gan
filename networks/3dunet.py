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

        self.decoder0 = self.decoder(base_n_filter*24, base_n_filter*8, ksize=4, pad1=1)
        self.decoder1 = self.decoder(base_n_filter*12, base_n_filter*4, ksize=5, pad1=0, pad2=2)
        self.decoder2 = self.decoder(base_n_filter*6, base_n_filter*2, ksize=7)

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

    def decoder(self, in_channels, out_channels, ksize, pad1=1, pad2=0): 
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels = in_channels, out_channels = out_channels, kernel_size = ksize, stride = 2, padding = pad1), 
            nn.ReLU(0.3),
            nn.ConvTranspose3d(in_channels = out_channels, out_channels = out_channels, kernel_size = ksize, stride = 1, padding=pad2), 
            nn.BatchNorm3d(num_features = out_channels),
            nn.ReLU(0.3)
        )
    
    def output(self,in_channels, out_channels): 
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 4, stride = 2, padding = 1), 
        )

    def print_size(self, x , name): 
        print('{} size: {}'.format(name, x.size()))

    def forward(self, x): 
        """
        Model execution 
        """
        self.print_size(x, 'x')
        down1 = self.encoder0(x)
        self.print_size(down1, 'down1')
        down2 = self.encoder1(down1)
        self.print_size(down2, 'down2')
        down3 = self.encoder2(down2)
        self.print_size(down3, 'down3')
       
        bot = self.bneck0(down3)
        bot = self.bneck1(bot)
        
        self.print_size(bot, 'final bneck')
        up1 = torch.cat([bot, down3], dim=1)
        self.print_size(up1, 'concat')
        up1 = self.decoder0(up1)	 
        self.print_size(up1, 'up1')
        
        up2 = torch.cat([up1, down2], dim=1)
        up2 = self.decoder1(up2)
        self.print_size(up2, 'up2')
        up3 = torch.cat([up2, down1], dim =1)
        up3 = self.decoder2(up3)
        self.print_size(up3, 'up3')

        out = self.output_(up3)
        # self.print_size(out, 'output')
        return out


    #     self.down_samp = nn.MaxPool3d(kernel_size = 2, stride = 2)
    #     self.encoder_1 = self.conv_norm_relu(in_channels, base_n_filter*2, block_type = 'encoder')
    #     self.encoder_2 = self.conv_norm_relu(base_n_filter*2, base_n_filter*4, block_type='encoder')
    #     self.encoder_3 = self.conv_norm_relu(base_n_filter*4, base_n_filter*8, block_type='encoder')
        
    #     self.bottleneck = self.conv_norm_relu(base_n_filter*8, base_n_filter*16, block_type='bottleneck')
        
    #     self.upscale = self.up_scale()
    #     self.up_samp_1 = self.up_conv(base_n_filter*16, ksize=5)
    #     # self.decoder_1 = self.conv_norm_relu(base_n_filter*24, base_n_filter*8, block_type='decoder')

    #     self.up_samp_2 = self.up_conv(base_n_filter*8, ksize=9)
    #     # self.decoder_2 = self.conv_norm_relu(base_n_filter*12, base_n_filter*4, block_type='decoder')

    #     self.up_samp_3 = self.up_conv(base_n_filter*4, ksize=1)
    #     # self.decoder_3 = self.conv_norm_relu(base_n_filter*6, base_n_filter*2, block_type='decoder')

    #     self.decoder_1 = self.deconv_norm_relu(base_n_filter*24, base_n_filter*8)
    #     self.decoder_2 = self.deconv_norm_relu(base_n_filter*12, base_n_filter*4)
    #     self.decoder_3 = self.deconv_norm_relu(base_n_filter*6, base_n_filter*2)

    #     self.output = nn.Conv3d(base_n_filter*2, num_classes, kernel_size=1)

    # def conv_norm_relu(self, feat_in, feat_out, block_type):
    #     types = {
    #         'encoder': 0.5,
    #         'bottleneck': 0.5,
    #         'decoder': 1
    #     }
    #     mid_layer_filter = int(feat_out* types.get(block_type))
        
    #     return nn.Sequential(
    #         nn.Conv3d(in_channels = feat_in, out_channels = mid_layer_filter, kernel_size = 3),
    # 		nn.BatchNorm3d(num_features= mid_layer_filter),
    # 		nn.ReLU(0.3), 
    #         nn.Conv3d(in_channels =  mid_layer_filter, out_channels = feat_out, kernel_size = 3),
    #         nn.BatchNorm3d(num_features = feat_out), 
    #         nn.ReLU(0.3)
    #     )
    
    # def deconv_norm_relu(self, feat_in, feat_out): 
    #     return nn.Sequential(
    #         nn.ConvTranspose3d(in_channels = feat_in, out_channels = feat_out, kernel_size = 3),
    # 		nn.BatchNorm3d(num_features= feat_out),
    # 		nn.ReLU(0.3), 
    #         nn.ConvTranspose3d(in_channels = feat_out, out_channels = feat_out, kernel_size = 3),
    #         nn.BatchNorm3d(num_features = feat_out), 
    #         nn.ReLU(0.3)
    #     )

    # def up_conv(self, n_filters, ksize):
    #     return nn.ConvTranspose3d(in_channels=n_filters, out_channels = n_filters, kernel_size=ksize)

    # def up_scale(self): 
    #     return nn.Upsample(scale_factor=2)

    # def forward(self, x):
    #     # encoder level 1
    #     out = self.encoder_1(x)
    #     concat_1 = out
    #     out = self.down_samp(out)
    #     print()
    #     print_size('concat_1', concat_1)

    #     # encoder level 2
    #     out = self.encoder_2(out)
    #     concat_2 =out
    #     out = self.down_samp(out)
    #     print_size('concat_2', concat_2)

         
    #     # encoder level 3
    #     out = self.encoder_3(out)
    #     concat_3 =out
    #     out = self.down_samp(out)
    #     print_size('concat_3', concat_3)


    #     # bottleneck
    #     out = self.bottleneck(out)
    #     print_size('bottle', out)

    #     # decoder level 1
    #     out = self.up_samp_1(out)
    #     # out = self.upscale(out)
    #     print_size('up_samp_1', out)
    #     out = torch.cat([out, concat_3], dim=1)
    #     out = self.decoder_1(out)
    #     print_size('deconv_1', out)

    #     # decoder level 2
    #     # out = self.up_samp_2(out)
    #     out = self.upscale(out)
    #     print_size('up_samp_2', out)
    #     out = torch.cat([out, concat_2], dim=1)
    #     out = self.decoder_2(out)
    #     print_size('decoder_2', out)

    #     # decoder level 3
    #     print_size('concat_1', concat_1)
    #     out = self.upscale(out)
    #     print_size('up_samp_3', out)
    #     out = torch.cat([out, concat_1], dim=1)
    #     out = self.decoder_3(out)

    #     return self.output(out)
