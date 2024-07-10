import pandas as pd
import h5py
from torch.utils.data import Dataset
from torchvision import transforms


class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		model_type=None,
		target_patch_size=-1,
		transform=None
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.wsi = wsi
		self.model_type = model_type
		self.transform = transform
		
		if not self.transform:
			if target_patch_size > 0:
				transform_list = [transforms.Resize((target_patch_size, target_patch_size))]
			else:
				transform_list = []
			transform_list.extend([
				transforms.ToTensor(),
				transforms.Normalize(
					mean=(0.485, 0.456, 0.406), 
					std=(0.229, 0.224, 0.225)
				),
			])
			self.transform = transforms.Compose(transform_list)
			print("Image processing: ", self.transform)
		
		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		if self.model_type == "plip":
			img = self.transform(images=img, return_tensors="pt")["pixel_values"]
		elif self.model_type == "ssl":
			img = self.transform(img, return_tensors="pt")["pixel_values"]
		else:
			img = self.transform(img).unsqueeze(0)
		return img, coord

class Dataset_All_Bags(Dataset):
	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]
