import os
import numpy as np
from PIL import Image
from config.config import cfg

root = cfg.DATA_PATH_SYSU

rgb_cameras = ['cam1','cam2','cam4','cam5']
ir_cameras  = ['cam3','cam6']

# load id info
file_path_train = os.path.join(root, 'exp/train_id.txt')
with open(file_path_train, 'r') as file:
	ids = file.read().splitlines()
	ids = [int(y) for y in ids[0].split(',')]
	id_train = ["%04d" % x for x in ids]

file_path_val = os.path.join(root, 'exp/val_id.txt')
with open(file_path_val, 'r') as file:
	ids = file.read().splitlines()
	ids = [int(y) for y in ids[0].split(',')]
	id_val = ["%04d" % x for x in ids]

# combine train and val split
id_train.extend(id_val)

img_paths_rgb = []
img_paths_ir  = []
for pid in sorted(id_train):
	for cam in rgb_cameras:
		img_dir = os.path.join(root, cam, pid)
		if os.path.isdir(img_dir):
			new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
			img_paths_rgb.extend(new_files)

	for cam in ir_cameras:
		img_dir = os.path.join(root, cam, pid)
		if os.path.isdir(img_dir):
			new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
			img_paths_ir.extend(new_files)

# relabel
pid_container = set()
for img_path in img_paths_ir:
	pid = int(img_path[-13:-9])
	pid_container.add(pid)

pid2label = {pid:label for label, pid in enumerate(pid_container)}

def read_imgs(img_paths, img_w, img_h):
	train_img = []
	train_label = []
	for img_path in img_paths:
		# img
		img = Image.open(img_path)
		img = img.resize((img_w, img_h), Image.ANTIALIAS)
		pix_array = np.array(img)
		train_img.append(pix_array)

		# label
		pid = int(img_path[-13:-9])
		label = pid2label[pid]
		train_label.append(label)

	return np.array(train_img), np.array(train_label)


train_img, train_label = read_imgs(img_paths_rgb, img_w=cfg.W, img_h=cfg.H)
np.save(os.path.join(root, 'train_rgb_resized_img.npy'), train_img)
np.save(os.path.join(root, 'train_rgb_resized_label.npy'), train_label)

train_img, train_label = read_imgs(img_paths_ir, img_w=cfg.W, img_h=cfg.H)
np.save(os.path.join(root, 'train_ir_resized_img.npy'), train_img)
np.save(os.path.join(root, 'train_ir_resized_label.npy'), train_label)
