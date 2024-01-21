import torch
import torch.nn as nn

from encoders import encoders
from decoders import decoders


class Encoder(nn.Module):
	def __init__(self, depth):
        super(Encoder, self).__init__()
        assert 1 <= depth <= 5

		self.depth = int(depth)
        self.model = encoders[depth+1]
        self.model.load_state_dict(torch.load(f'models/encoders/encoder{self.depth}.pth'))

	def forward(self, x):
		return self.model(x)


class Decoder(nn.Module):
	def __init__(self, depth):
        super(Decoder, self).__init__()
        assert 1 <= depth <= 5

        self.depth = int(depth)
        self.model = decoders[depth+1]
        self.model.load_state_dict(torch.load(f'models/encoders/encoder{self.depth}.pth'))

	def forward(self, x):
		return self.model(x)


