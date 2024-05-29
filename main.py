import os
import time
import h5py
import openslide
import argparse
import shutil
import torch

from datasets import Dataset_All_Bags
from utils import feature_extraction, init_model, print_network

# This code is adapted from CLAM by Mahmoodlab (https://github.com/mahmoodlab/CLAM)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_root_dir', type=str, default="/media/nfs/SURV/TCGA_OV/")
parser.add_argument('--data_h5_dir', type=str, default="SP1024")
parser.add_argument('--data_slide_dir', type=str, default="Slides")
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_file', type=str, default="slide_list.csv")
parser.add_argument('--feat_dir', type=str, default="Feats1024")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--model_type', type=str, default="ctp")
parser.add_argument('--ckpt_path', type=str, default='./models/ctranspath.pth')
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

	tmp_folder = os.path.join(args.data_root_dir, 'tmp', args.model_type.upper())
	os.makedirs(tmp_folder, exist_ok=True)

	args.csv_path = os.path.join(args.data_root_dir, args.csv_file)
	
	if not os.path.isfile(args.csv_path):
		raise NotImplementedError

	print('loading model checkpoint')
	processor = None
	if args.model_type == "conch":
		from conch.open_clip_custom import create_model_from_pretrained
		model, processor = create_model_from_pretrained('conch_ViT-B-16', args.ckpt_path)
	else:
		model = init_model(args)
		if args.model_type == "ssl":
			from transformers import AutoImageProcessor, CLIPProcessor
			processor = AutoImageProcessor.from_pretrained("owkin/phikon")
		elif args.model_type == "plip":
			from transformers import CLIPProcessor
			processor = CLIPProcessor.from_pretrained("vinid/plip")
	
	model.to(device)
	print_network(model)
	if torch.cuda.device_count() > 1:
		model = torch.nn.DataParallel(model)
	model.eval()

	bags_dataset = Dataset_All_Bags(args.csv_path)
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
		output_file_path = feature_extraction(args,
			h5_file_path, tmp_file_path, wsi, 
			model, processor,
			verbose = 1, print_every = 20
		)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, bag_base+'.pt'))
	
	print("You may delete the tmp folder in the root_dir.")