# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

import torchvision.transforms as transforms
import torchvision.datasets as dset
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import (save_net,
								load_net,
								vis_detections,
								vis_detections_PIL,
								vis_detections_filtered_objects_PIL,
								vis_detections_filtered_objects)
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

import pathlib
from collections import deque

STATE_MAP = {0: 'No Contact', 1: 'Self Contact', 2: 'Another Person', 3: 'Portable Object', 4: 'Stationary Object'}
pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
model_path = 'models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth'

def parse_args():
	"""Parse input arguments"""
	parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
	parser.add_argument('--dataset', default='pascal_voc', type=str, help='training dataset')
	parser.add_argument('--cfg', dest='cfg_file', default='cfgs/res101.yml', type=str, help='optional config file')
	parser.add_argument('--net', default='res101', type=str, help='vgg16, res50, res101, res152')
	parser.add_argument('--set', dest='set_cfgs', nargs=argparse.REMAINDER, help='set config keys')
	parser.add_argument('--vid_path', help='path to input video file')
	parser.add_argument('--save_dir', default="out", help='directory to save results')
	parser.add_argument('--cuda', action='store_true', help='whether use CUDA')
	parser.add_argument('--mGPUs', action='store_true', help='whether use multiple GPUs')
	parser.add_argument('--cag', dest='class_agnostic', action='store_true', help='whether perform class_agnostic bbox regression')
	parser.add_argument('--parallel_type', default=0, type=int, help='which part of model to parallel, 0: all, 1: model before roi pooling')
	parser.add_argument('--checksession', default=1, type=int, help='checksession to load model')
	parser.add_argument('--checkepoch', default=8, type=int, help='checkepoch to load network')
	parser.add_argument('--checkpoint', default=132028, type=int, help='checkpoint to load network')
	parser.add_argument('--bs', default=1, type=int, help='batch_size')
	parser.add_argument('--vis', default=True, help='visualization mode')
	parser.add_argument('--webcam_num', default=-1, type=int, help='webcam ID number')
	parser.add_argument('--thresh_hand', type=float, default=0.5)
	parser.add_argument('--thresh_obj', type=float, default=0.5)
	parser.add_argument('--buffer_size', default=5, type=int)
	parser.add_argument('--top_k', default=1, type=int)
	parser.add_argument('--thresh_dist', default=500, type=float)

	return parser.parse_args()

def _get_image_blob(im):
	"""Converts an image into a network input."""
	im_orig = im.astype(np.float32, copy=True)
	im_orig -= cfg.PIXEL_MEANS

	im_shape = im_orig.shape
	im_size_min = np.min(im_shape[0:2])
	im_size_max = np.max(im_shape[0:2])

	processed_ims = []
	im_scale_factors = []

	for target_size in cfg.TEST.SCALES:
		im_scale = float(target_size) / float(im_size_min)
		if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
			im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
		im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
		im_scale_factors.append(im_scale)
		processed_ims.append(im)

	blob = im_list_to_blob(processed_ims)
	return blob, np.array(im_scale_factors)

def load_model(args):
	"""Load Faster R-CNN model."""	
	if args.net == 'vgg16':
		fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
	elif args.net.startswith('res'):
		num_layers = int(args.net[3:])
		fasterRCNN = resnet(pascal_classes, num_layers, pretrained=False, class_agnostic=args.class_agnostic)
	else:
		raise RuntimeError("model type not recognized.")

	fasterRCNN.create_architecture()
 
	print(f"loading checkpoint {model_path}...")
	checkpoint = torch.load(model_path) if args.cuda else torch.load(model_path, map_location=lambda storage, loc: storage)
	fasterRCNN.load_state_dict(checkpoint['model'])

	if 'pooling_mode' in checkpoint.keys():
		cfg.POOLING_MODE = checkpoint['pooling_mode']

	if args.cuda:
		fasterRCNN.cuda()

	fasterRCNN.eval()
 
	print('loaded model successfully!')
 
	return fasterRCNN

def prepare_tensors(args):
	"""Prepare tensors for input data."""
	im_data = torch.FloatTensor(1)
	im_info = torch.FloatTensor(1)
	num_boxes = torch.LongTensor(1)
	gt_boxes = torch.FloatTensor(1)
	box_info = torch.FloatTensor(1)

	if args.cuda:
		im_data = im_data.cuda()
		im_info = im_info.cuda()
		num_boxes = num_boxes.cuda()
		gt_boxes = gt_boxes.cuda()

	return im_data, im_info, num_boxes, gt_boxes, box_info

def initialize_video_capture(args):
	"""Initialize video capture."""
	if args.webcam_num >= 0:
		cap = cv2.VideoCapture(args.webcam_num)
		if not cap.isOpened():
			raise RuntimeError("Webcam could not open. Please check connection.")
		print("using webcam")
	else:
		cap = cv2.VideoCapture(args.vid_path)
		if not cap.isOpened():
			raise RuntimeError("Could not open video.")
		print(f'using video at {args.vid_path}')
		print(f'save dir = {args.save_dir}')
	return cap

def process_frame(args, frame, fasterRCNN, im_data, im_info, num_boxes, gt_boxes, box_info):
	"""Process a single video frame."""
	im_in = np.array(frame)
	im = im_in

	blobs, im_scales = _get_image_blob(im)
	assert len(im_scales) == 1, "Only single-image batch implemented"
	im_blob = blobs
	im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

	im_data_pt = torch.from_numpy(im_blob).permute(0, 3, 1, 2)
	im_info_pt = torch.from_numpy(im_info_np)

	# with torch.no_grad():
	im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
	im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
	gt_boxes.resize_(1, 1, 5).zero_()
	num_boxes.resize_(1).zero_()
	box_info.resize_(1, 1, 5).zero_() 

	# pdb.set_trace()
 
	rois, cls_prob, bbox_pred, \
	rpn_loss_cls, rpn_loss_box, \
	RCNN_loss_cls, RCNN_loss_bbox, \
	rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info) 

	scores = cls_prob.data
	boxes = rois.data[:, :, 1:5]

	# extract predicted params
	contact_vector = loss_list[0][0] # hand contact state info
	offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
	lr_vector = loss_list[2][0].detach() # hand side info (left/right)

	# get hand contact 
	_, contact_indices = torch.max(contact_vector, 2)
	contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

	# get hand side 
	lr = (torch.sigmoid(lr_vector) > 0.5).squeeze(0).float()

	if cfg.TEST.BBOX_REG:
		# Apply bounding-box regression deltas
		box_deltas = bbox_pred.data
		if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
		# Optionally normalize targets by a precomputed mean and stdev
			if args.class_agnostic:
				if args.cuda > 0:
					box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
							+ torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
				else:
					box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
							+ torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

				box_deltas = box_deltas.view(1, -1, 4)
			else:
				if args.cuda > 0:
					box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
							+ torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
				else:
					box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
							+ torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
				box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

		pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
		pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
	else:
		# Simply repeat the boxes, once for each class
		pred_boxes = np.tile(boxes, (1, scores.shape[1]))

	pred_boxes /= im_scales[0]

	scores = scores.squeeze()
	pred_boxes = pred_boxes.squeeze()

	return scores, pred_boxes, contact_indices, offset_vector, lr

def detect_objects(args, scores, pred_boxes, contact_indices, offset_vector, lr):
	"""Detect objects in the frame."""
	obj_dets = hand_dets = all_dets = None
 
	for j in range(1, len(pascal_classes)):
		# thresholding, optionally based on class
		# inds = torch.nonzero(scores[:,j] > thresh).view(-1) # use same threshold for all classes
		if pascal_classes[j] == 'hand':
			inds = torch.nonzero(scores[:,j] > args.thresh_hand).view(-1)
		elif pascal_classes[j] == 'targetobject':
			inds = torch.nonzero(scores[:,j] > args.thresh_obj).view(-1)

		# if there is det
		if inds.numel() > 0:
			cls_scores = scores[:,j][inds]
			_, order = torch.sort(cls_scores, 0, True)
			if args.class_agnostic:
				cls_boxes = pred_boxes[inds, :]
			else:
				cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
			
			cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
			cls_dets = cls_dets[order]
			keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
			cls_dets = cls_dets[keep.view(-1).long()]
			if pascal_classes[j] == 'targetobject':
				obj_dets = cls_dets.cpu().numpy()
			elif pascal_classes[j] == 'hand':
				hand_dets = cls_dets.cpu().numpy()
	
			if all_dets is None: # if we want to visualize all classes (inc. background), not just hand + object
				all_dets = cls_dets.cpu().numpy()
			else:
				all_dets = np.concatenate((all_dets, cls_dets.cpu().numpy()), 0)
	 
	return obj_dets, hand_dets, all_dets


def get_new_contact_vals(hand_dets, thresh_hand):
	new_vals = [False, False] # whether contact is detected in current frame
	
	if hand_dets is not None:
		for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))):
			bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
			score = hand_dets[i, 4]
			lr = hand_dets[i, -1]
			# print(lr)
			state = hand_dets[i, 5]
			new_vals[int(lr)] = (score > thresh_hand and state == 3) # condition: hand detected and contacting portable object

	return new_vals

def update_buffers(detection_buffers, curr_detected, frame_buffer, new_vals, save_dir):
	# fancy debounce
	
	for i,buffer in enumerate(detection_buffers):
		lr_name = "left" if i == 0 else "right"
		buffer.append(new_vals[i])

		if all(buffer) and not curr_detected[i]:
			print(f"detected {lr_name} start contact, stable")
			condition = f"contact_{lr_name}"
		elif not any(buffer) and curr_detected[i]:
			print(f"detected {lr_name} stop contact, stable")
			condition = f"no_contact_{lr_name}"
		else:
			continue

		curr_detected[i] = not curr_detected[i]
		
		old_frame = frame_buffer[0]

		if old_frame is not None:
			os.makedirs(save_dir, exist_ok=True)
			result_path = os.path.join(save_dir, f'frame_{int(time.time())}_{condition}.png')
			old_frame.save(result_path)
			print(f"saved frame to {result_path}")
		else:
			print("just starting")

def main():
	args = parse_args()

	if args.cfg_file is not None:
		cfg_from_file(args.cfg_file)
	if args.set_cfgs is not None:
		cfg_from_list(args.set_cfgs)

	buffer_size = args.buffer_size
	thresh_hand = args.thresh_hand
	thresh_obj = args.thresh_obj
	top_k = args.top_k
	vis = args.vis
	save_dir = args.save_dir

	cfg.USE_GPU_NMS = args.cuda
	np.random.seed(cfg.RNG_SEED)

	fasterRCNN = load_model(args)
	im_data, im_info, num_boxes, gt_boxes, box_info = prepare_tensors(args)
	cap = initialize_video_capture(args)

	detection_buffers = [
		deque(buffer_size * [False], buffer_size),
		deque(buffer_size * [False], buffer_size)
	]
	curr_detected = [False, False]
	frame_buffer = deque(buffer_size * [None], buffer_size)

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		with torch.no_grad():
			scores, pred_boxes, contact_indices, offset_vector, lr = process_frame(args, frame, fasterRCNN, im_data, im_info, num_boxes, gt_boxes, box_info)
			obj_dets, hand_dets, all_dets = detect_objects(args, scores, pred_boxes, contact_indices, offset_vector, lr)
		
		im2show = np.copy(frame)
		
		if vis:
			im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj, top_k=top_k)
		
		frame_buffer.append(im2show)
		new_vals = get_new_contact_vals(hand_dets, thresh_hand)
		
		update_buffers(detection_buffers, curr_detected, frame_buffer, new_vals, save_dir)

		im2show = np.array(im2show)
		im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
		cv2.imshow("frame", im2showRGB)

		if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
			break

	print("done, exiting...")

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
