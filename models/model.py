import torch
import torch.nn as nn
from torchvision import transforms

from models.encoders import encoders
from models.decoders import decoders

from utils import whitening_coloring_transform, DEFAULT_ENCODER_DIR, DEFAULT_DECODER_DIR


class Encoder(nn.Module):
	def __init__(self, depth, load_weights=True, checkpoint_path=None):
		super(Encoder, self).__init__()
		self.depth = depth
		self.model = encoders[depth-1]
		self.checkpoint_path = checkpoint_path if checkpoint_path else DEFAULT_ENCODER_DIR + f'{self.depth}.pth'

		if load_weights:
			state_dict = torch.load(self.checkpoint_path)
			state_dict = {'.'.join(k.split('.')[-2:]): v for k, v in state_dict.items()}			
			self.model.load_state_dict(state_dict)
			print(f'Loaded encoder checkpoint at: {self.checkpoint_path}')

	def forward(self, x):
		return self.model(x)


class Decoder(nn.Module):
	def __init__(self, depth, load_weights=True, checkpoint_path=None):
		super(Decoder, self).__init__()
		self.depth = depth
		self.model = decoders[depth-1]
		self.checkpoint_path = checkpoint_path if checkpoint_path else DEFAULT_DECODER_DIR + f'{self.depth}.pth'

		if load_weights:
			state_dict = torch.load(self.checkpoint_path)
			state_dict = {'.'.join(k.split('.')[-2:]): v for k, v in state_dict.items()}
			self.model.load_state_dict(state_dict)
			print(f'Loaded decoder checkpoint at: {self.checkpoint_path}')

	def forward(self, x):
		return self.model(x)


class StylizationModel(nn.Module):
	def __init__(self, level='single', strength=1., depth=4, device='cpu', decoder_checkpoint_path=None):
		super(StylizationModel, self).__init__()

		if strength:
			assert 0 <= strength <= 1, 'Stylization strength should be in the range [0, 1].'
			self.strength = strength
		elif level == 'single':
			self.strength = 1.
		else:
			self.strength = 0.2 * depth

		self.device = device # advised to do SVD on CPU
		if level == 'single':
			self.encoders = [Encoder(depth)]
			self.decoders = [Decoder(depth, checkpoint_path=decoder_checkpoint_path)]
		else:
			self.encoders = [Encoder(d, checkpoint_path=decoder_checkpoint_path) for d in range(depth, 0, -1)]
			self.decoders = [Decoder(d, checkpoint_path=decoder_checkpoint_path) for d in range(depth, 0, -1)]

	def forward(self, content, style):
		device = content.device
		for depth in range(len(self.encoders)):
			content_feature = self.encoders[depth](content).data.to(device=self.device)
			style_feature = self.encoders[depth](style).data.to(device=self.device)
			stylized_feature = whitening_coloring_transform(content_feature.squeeze(0), style_feature.squeeze(0), strength=self.strength)
			content = self.decoders[depth](stylized_feature.data.to(device).unsqueeze(0))
			content = torch.clamp(content, min=0., max=1.)
		return content.squeeze(0)

