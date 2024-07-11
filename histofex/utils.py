import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from histofex.datasets import Whole_Slide_Bag_FP

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def init_model(model_type, ckpt_path):
	print('loading model checkpoint')
	processor = None
	if model_type == "conch":
		from conch.open_clip_custom import create_model_from_pretrained
		model, processor = create_model_from_pretrained('conch_ViT-B-16', ckpt_path)

	elif model_type == "resnet50":
		model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
		model.fc = nn.Identity()
	
	elif model_type == "ssl":
		from transformers import ViTModel, AutoImageProcessor
		model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
		processor = AutoImageProcessor.from_pretrained("owkin/phikon")
		
	elif model_type == "ctp":
		from histofeatex.ctran import ctranspath
		model = ctranspath()
		model.head = nn.Identity()
		model.load_state_dict(torch.load(ckpt_path)["model"], strict=True)
		
	elif model_type == "plip":
		from transformers import CLIPModel, CLIPProcessor
		model = CLIPModel.from_pretrained("vinid/plip")
		processor = CLIPProcessor.from_pretrained("vinid/plip")
	
	elif model_type == "uni":
		import timm
		model = timm.create_model(
			"vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
		)
		model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
	
	
	model.to(device)
	print_network(model)
	if torch.cuda.device_count() > 1:
		model = torch.nn.DataParallel(model)
	model.eval()
	return model, processor

def feature_extraction(args, file_path, output_path, wsi, model, processor,
 	verbose = 0, print_every=20):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		verbose: level of feedback
	"""

	dataset = Whole_Slide_Bag_FP(
		file_path=file_path, wsi=wsi, model_type=args.model_type, 
		target_patch_size=args.target_patch_size, transform=processor
	)
	
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * args.batch_size))
			batch = batch.to(device, non_blocking=True)
			
			if args.model_type == "plip":
				features = model.get_image_features(batch)
			elif args.model_type == "conch":
				features = model.encode_image(batch, proj_contrast=False, normalize=False)
			else:
				features = model(batch)

			if args.model_type == "ssl":
				features = features.last_hidden_state[:, 0, :]
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


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

