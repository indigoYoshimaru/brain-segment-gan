import torch.nn as nn
import torch
from utils.run_util import print_size

class MultiscaleGenerator(nn.Module):
    def __init__(self, in_channels, num_classes, base_n_filter = 8):
        super(MultiscaleGenerator, self).__init__()
        
        self.outputs = []
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(in_channels, base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(base_n_filter, base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(base_n_filter, base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(base_n_filter)

        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(base_n_filter, base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(base_n_filter*2, base_n_filter*2)
        self.inorm3d_c2 = nn.InstanceNorm3d(base_n_filter*2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(base_n_filter*2, base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(base_n_filter*4, base_n_filter*4)
        self.inorm3d_c3 = nn.InstanceNorm3d(base_n_filter*4)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(base_n_filter*4, base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(base_n_filter*8, base_n_filter*8)
        self.inorm3d_c4 = nn.InstanceNorm3d(base_n_filter*8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(base_n_filter*8, base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(base_n_filter*16, base_n_filter*16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(base_n_filter*16, base_n_filter*8)

        self.conv3d_l0 = nn.Conv3d(base_n_filter*8, base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(base_n_filter*8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(base_n_filter*16, base_n_filter*16)
        self.conv3d_l1 = nn.Conv3d(base_n_filter*16, base_n_filter*8, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(base_n_filter*8, base_n_filter*4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(base_n_filter*8, base_n_filter*8)
        self.conv3d_l2 = nn.Conv3d(base_n_filter*8, base_n_filter*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(base_n_filter*4, base_n_filter*2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(base_n_filter*4, base_n_filter*4)
        self.conv3d_l3 = nn.Conv3d(base_n_filter*4, base_n_filter*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(base_n_filter*2, base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(base_n_filter*2, base_n_filter*2)
        self.conv3d_l4 = nn.Conv3d(base_n_filter*2, num_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.ds1_1x1_conv3d = nn.Conv3d(base_n_filter*16, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds2_1x1_conv3d = nn.Conv3d(base_n_filter*8, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(base_n_filter*4, num_classes, kernel_size=1, stride=1, padding=0, bias=False)


    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        #  Level 1 context 
        # print_size('x', x)
        out = self.conv3d_c1_1(x)
        residual_1 = out
        ##print_size('en-res1', residual_1 )
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)
        context_1 = out
        ##print_size('en-out1', out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        #print_size('en-res2', residual_2 )
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out
        #print_size('en-out2', out)


        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        #print_size('en-res3', residual_3 )
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out
        #print_size('en-out3', out)

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        #print_size('en-res4', residual_4)
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out
        #print_size('en-out4', out)


        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        #print_size('en-res5', residual_5 )
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        # print_size('en-out5', out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)
        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)
        #print_size('en-out5', out)
        #print('===========================')

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        ds1 = out 
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)
        #print_size('de-scale1', ds1)
        #print_size('de-out1', out)

        # Level 2 localization pathway
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)
        #print_size('de-scale2', ds2)
        #print_size('de-out2', out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)
        #print_size('de-scale3', ds3)
        #print_size('de-out3', out)

        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)
        #print_size('de-out4', out)
    
        # output 
        ds1_conv = self.ds1_1x1_conv3d(ds1)
        ds1_upscale = self.upscale(ds1_conv)
        ds2_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum = ds1_upscale + ds2_conv
        ds1_ds2_sum_upscale = self.upscale(ds1_ds2_sum)
        ds3_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upscale(ds1_ds2_sum_upscale_ds3_sum)
        #print_size('ds1-conv', ds1_conv)
        #print_size('ds2-conv', ds2_conv)
        #print_size('ds3-conv', ds3_conv)

        final_out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        #print_size('final', final_out)
        #print('===========================')

        return final_out
