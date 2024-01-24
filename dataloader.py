import os
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

class UnpairedDataset(Dataset):
	def __init__(self, root_dir, max_side=768, max_data=None):
		super(UnpairedDataset, self).__init__()
		self.root_dir = root_dir
		self.max_side = max_side
		self.images = os.listdir(root_dir)
		if max_data:
			self.images = np.random.choice(self.images, size=max_data, replace=False)
		self.transform = transforms.Compose([
			transforms.Lambda(lambda x: self._resize_and_pad(x)),
			transforms.ToTensor(),
		])

	def _resize_and_pad(self, image):
		w, h = image.size
		aspect_ratio = w / h
		if w >= h:
			new_w = self.max_side
			new_h = int(self.max_side / aspect_ratio)
		else:
			new_h = self.max_side
			new_w = int(self.max_side * aspect_ratio)
		image = image.resize((new_w, new_h))

		pad_width = self.max_side - new_w
		pad_height = self.max_side - new_h
		padding = (pad_width // 2, pad_height // 2, pad_width - (pad_width // 2), pad_height - (pad_height // 2))
		image = transforms.functional.pad(image, padding, fill=0, padding_mode='constant')

		return image

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		img_path = os.path.join(self.root_dir, self.images[idx])
		image = Image.open(img_path).convert('RGB')
		image = self._resize_and_pad(image)
		if self.transform:
			image = self.transform(image)
		return image


class PairedDataset(Dataset):
	def __init__(self, content_dir, style_dir):
		super(PairedDataset, self).__init__()

		self.content_dir = content_dir
		self.style_dir = style_dir
		self.pairs = []

		print('Creating paired dataset...')
		for style in tqdm(os.listdir(style_dir)):
			for content in os.listdir(content_dir):
				self.pairs.append({'content': content, 'style': style})

		self.transform = transforms.Compose([
			transforms.ToTensor(),
		])

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, idx):
		content_name = self.pairs[idx]['content']
		content_path = os.path.join(self.content_dir, content_name)
		content = Image.open(content_path).convert('RGB')
		
		style_name = self.pairs[idx]['style']
		style_path = os.path.join(self.style_dir, style_name)
		style = Image.open(style_path).convert('RGB')
		
		if self.transform:
			content = self.transform(content)
			style = self.transform(style)

		content_name = '.'.join(content_name.split('.')[:-1])
		style_name = '.'.join(style_name.split('.')[:-1])
		return {'content': content, 'style': style, 'content_name': content_name, 'style_name': style_name}


def get_dataloader(root_dir, batch_size, max_side=768, is_validation=False, max_data=None):
	dataset = UnpairedDataset(root_dir=root_dir, max_side=max_side, max_data=max_data)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=not is_validation)
	return dataloader

def get_stylization_dataloader(content_dir, style_dir):
	dataset = PairedDataset(content_dir=content_dir, style_dir=style_dir)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
	return dataloader