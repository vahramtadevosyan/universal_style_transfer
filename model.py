import torch
import torch.nn as nn

from encoders import encoders
from decoders import decoders


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
			state_dict = torch.load(f'checkpoints/decoder{self.depth}.pth')
			state_dict = {'.'.join(k.split('.')[-2:]): v for k, v in state_dict.items()}
			self.model.load_state_dict(state_dict)

	def forward(self, x):
		return self.model(x)

