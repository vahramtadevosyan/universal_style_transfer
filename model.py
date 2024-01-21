import torch
import torch.nn as nn

from encoders import encoders
from decoders import decoders


class Encoder(nn.Module):
	def __init__(self, depth):
		super(Encoder, self).__init__()
		assert 1 <= depth <= 5
		self.depth = int(depth)
		self.model = encoders[depth-1]
		
		state_dict = torch.load(f'models/encoders/encoder{self.depth}.pth')
		self.model.load_state_dict(state_dict)

	def forward(self, x):
		return self.model(x)


class Decoder(nn.Module):
	def __init__(self, depth):
		super(Decoder, self).__init__()
		assert 1 <= depth <= 5
		self.depth = int(depth)
		self.model = decoders[depth-1]
		state_dict = torch.load(f'models/decoders/decoder{self.depth}.pth')
		self.model.load_state_dict(state_dict)

	def forward(self, x):
		return self.model(x)

