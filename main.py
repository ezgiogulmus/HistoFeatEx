import os
import time
import h5py
import openslide
import argparse
import shutil
import torch
from torch.utils.data import DataLoader

from datasets import Dataset_All_Bags, Whole_Slide_Bag_FP
from utils import save_hdf5, init_model, print_network, collate_features

# This code is adapted from CLAM by Mahmoodlab (https://github.com/mahmoodlab/CLAM)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1, model_type=None):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, model_type=model_type,
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			
			if model_type == "plip":
				features = model.get_image_features(batch)
			else:
				features = model(batch)

			if model_type == "ssl":
				features = features.last_hidden_state[:, 0, :]
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_root_dir', type=str, default="/media/nfs/SURV/TCGA_OV/")
parser.add_argument('--data_h5_dir', type=str, default="SP1024")
parser.add_argument('--data_slide_dir', type=str, default="Slides")
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default="slide_list.csv")
parser.add_argument('--feat_dir', type=str, default="Feats1024")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--model_type', type=str, default="ctp")
parser.add_argument('--ckpt_path', type=str, default='../LN/TransPath/ctranspath.pth')
args = parser.parse_args()


if __name__ == '__main__':
	print("################## SETTINGS ###################")
	print(vars(args))
	print("###############################################")
	print('initializing dataset')

	args.data_h5_dir = os.path.join(args.data_root_dir, args.data_h5_dir)
	args.data_slide_dir = os.path.join(args.data_root_dir, args.data_slide_dir)
	args.feat_dir = os.path.join(args.data_root_dir, args.feat_dir, args.model_type.upper())
	os.makedirs(args.feat_dir, exist_ok=True)
	dest_files = os.listdir(args.feat_dir)
	tmp_folder = os.path.join(args.data_root_dir, 'tmp')
	os.makedirs(tmp_folder)

	args.csv_path = os.path.join(args.data_root_dir, args.csv_path)
	
	if not os.path.isfile(args.csv_path):
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(args.csv_path)

	print('loading model checkpoint')
	model = init_model(args)
	model.to(device)
	
	print_network(model)
	if torch.cuda.device_count() > 1:
		model = torch.nn.DataParallel(model)
		
	model.eval()
	total = len(bags_dataset)

	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		tmp_file_path = os.path.join(tmp_folder, bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		output_file_path = compute_w_loader(h5_file_path, tmp_file_path, wsi, 
		model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
		custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size, 
		model_type=args.model_type)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, bag_base+'.pt'))
	
	shutil.rmtree(tmp_folder)

