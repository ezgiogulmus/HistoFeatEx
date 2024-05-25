import h5py
import numpy as np
import torch
from torch import nn
from torchvision import models


def init_model(args):
	if args.model_type == "resnet50":
		backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
		backbone.fc = nn.Identity()
		
	elif args.model_type == "ssl":
		from transformers import ViTModel
		backbone = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
		
	elif args.model_type == "ctp":
		from ctran import ctranspath
		backbone = ctranspath()
		backbone.head = nn.Identity()
		backbone.load_state_dict(torch.load(args.ckpt_path)["model"], strict=True)
		
	elif args.model_type == "plip":
		from transformers import CLIPModel
		backbone = CLIPModel.from_pretrained("vinid/plip")

	return backbone

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
	file = h5py.File(output_path, mode)
	for key, val in asset_dict.items():
		data_shape = val.shape
		if key not in file:
			data_type = val.dtype
			chunk_shape = (1, ) + data_shape[1:]
			maxshape = (None, ) + data_shape[1:]
			dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
			dset[:] = val
			if attr_dict is not None:
				if key in attr_dict.keys():
					for attr_key, attr_val in attr_dict[key].items():
						dset.attrs[attr_key] = attr_val
		else:
			dset = file[key]
			dset.resize(len(dset) + data_shape[0], axis=0)
			dset[-data_shape[0]:] = val
	file.close()
	return output_path

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)

