import torch
import torch.nn as nn

from encoders import encoders
from decoders import decoders

from wc_transform import whitening_coloring_transform

class Encoder(nn.Module):
	def __init__(self, depth, load_weights=True):
		super(Encoder, self).__init__()
		assert 1 <= depth <= 5
		self.depth = int(depth)
		self.model = encoders[depth-1]
		if load_weights:
			state_dict = torch.load(f'models/encoders/encoder{self.depth}.pth')
			state_dict = {'.'.join(k.split('.')[-2:]): v for k, v in state_dict.items()}			
			self.model.load_state_dict(state_dict)

	def forward(self, x):
		return self.model(x)


class Decoder(nn.Module):
	def __init__(self, depth, load_weights=True):
		super(Decoder, self).__init__()
		assert 1 <= depth <= 5
		self.depth = int(depth)
		self.model = decoders[depth-1]
		if load_weights:
			# state_dict = torch.load(f'checkpoints/decoder{self.depth}.pth')
			state_dict = torch.load(f'models/decoders/decoder{self.depth}.pth')
			state_dict = {'.'.join(k.split('.')[-2:]): v for k, v in state_dict.items()}
			self.model.load_state_dict(state_dict)

	def forward(self, x):
		return self.model(x)


class StylizationModel(nn.Module):
	def __init__(self, level='single', strength=1., depth=4, device='cpu'):
		super(StylizationModel, self).__init__()
		assert 0 <= strength <= 1, 'Stylization strength should be in the range [0, 1].'
		self.strength = strength
		self.device = device # advised to do SVD on CPU
		if level == 'single':
			assert 1 <= depth <= 5
			self.encoders = [Encoder(depth)]
			self.decoders = [Decoder(depth)]
		else:
			self.encoders = [Encoder(d) for d in range(len(encoders), 0, -1)]
			self.decoders = [Decoder(d) for d in range(len(decoders), 0, -1)]

	def forward(self, content, style):
		device = content.device
		content_feature = self.encoders[0](content).data.to(device=self.device)
		for depth in range(self.encoders):
			style_feature = self.encoders[depth](style).data.to(device=self.device)
			stylized_feature = whitening_coloring_transform(content_feature.squeeze(0), style_feature.squeeze(0), strength=self.strength)
			content_feature = self.decoders[depth](stylized_feature.data.to(device).unsqueeze(0))
		return content_feature


