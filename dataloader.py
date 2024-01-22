import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

class UnpairedDataset(Dataset):
	def __init__(self, root_dir, max_side=768):
		self.root_dir = root_dir
		self.max_side = max_side
		self.image_folder = datasets.ImageFolder(root=self.root_dir)
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
		return len(self.image_folder)

	def __getitem__(self, idx):
		image, _ = self.image_folder[idx]
		image = self._resize_and_pad(image)
		if self.transform:
			image = self.transform(image)
		return image

def get_dataloader(root_dir, batch_size, max_side=768, is_validation=False):
	dataset = UnpairedDataset(root_dir=root_dir, max_side=max_side)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=not is_validation)
	return dataloader

