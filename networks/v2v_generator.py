import torch
import torch.nn as nn
import torch.optim

class Generator(nn.Module): 
	"""
	Generator in UNet structure
	"""
	def __init__(self, in_channels, num_classes): 
		super().__init__()
		self.in_channels = in_channels
		self.num_classes = num_classes

		self.lrelu = nn.LeakyReLU()
		self.dropout3d = nn.Dropout3d(p=0.6)
		self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
		self.softmax = nn.Softmax(dim=1)
		
		# encoder
		
		self.encoder0 = self.encoder(self.in_channels, 64)
		self.encoder1 = self.encoder(64, 128)
		self.encoder2 = self.encoder(128, 256)
		self.encoder3 = self.encoder(256, 512)	

		self.bneck0 = self.bottleneck(512, 512)
		self.bneck1 = self.bottleneck(1024, 512)

		self.decoder0 = self.decoder(512, 256)
		self.decoder1 = self.decoder(512, 128)
		self.decoder2 = self.decoder(256, 64)

		self.output_ = self.output(128, self.num_classes)

	def encoder(self, in_channels, out_channels): 
		return nn.Sequential(
			nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 4, stride=2, padding = 1 ),
			nn.InstanceNorm3d(num_features= out_channels),
			nn.LeakyReLU(0.3)
		)

	def bottleneck(self, in_channels, out_channels): 
		return nn.Sequential(
			nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 4, stride = 1, padding = 'same'),
			nn.InstanceNorm3d(num_features = out_channels), 
			nn.LeakyReLU(0.3), 
			nn.Dropout3d(p = 0.2)
		)

	def decoder(self, in_channels, out_channels): 
		return nn.Sequential(
			nn.ConvTranspose3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 4, stride = 2, padding = 1), 
			nn.InstanceNorm3d(num_features = out_channels),
			nn.LeakyReLU(0.3)
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
		# self.print_size(x, 'x')
		down1 = self.encoder0(x)
		# self.print_size(down1, 'down1')
		down2 = self.encoder1(down1)
		# self.print_size(down2, 'down2')
		down3 = self.encoder2(down2)
		# self.print_size(down3, 'down3')
		down4 = self.encoder3(down3)
		# self.print_size(down4, 'down4')
		bot=down4
		bot_new = self.bneck0(bot) # 512-> 512
		for i in range(3): 
			bot_res = torch.cat([bot_new, bot], dim=1)
			bot = bot_new
			bot_new = self.bneck1(bot_res)
		
		# self.print_size(bot_new, 'final bneck')
		up1 = self.decoder0(bot_new)	 
		up1 = torch.cat([up1, down3], dim=1)
		# self.print_size(up1, 'up1')
		up2 = self.decoder1(up1)
		up2 = torch.cat([up2, down2], dim=1)
		# self.print_size(up2, 'up2')
		up3 = self.decoder2(up2)
		up3 = torch.cat([up3, down1], dim =1)
		# self.print_size(up3, 'up3')

		out = self.output_(up3)
		# self.print_size(out, 'output')
		return out
		
