import os
import shutil
from random import shuffle

train_val_ratio = 0.90

def create_binary_class_imagefolder():
	data_path = "data/images" # Original folder with all images
	train_path = "data/catdog/train" # Has to exist before running
	val_path = "data/catdog/val" # Has to exist before running

	# Get all file names and shuffle
	onlyfiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
	shuffle(onlyfiles)
	train_files = onlyfiles[:int(train_val_ratio * len(onlyfiles))]
	val_files = onlyfiles[int(train_val_ratio * len(onlyfiles)):]

	# Move to train and val folders
	for f in train_files:
		new_path = train_path
		new_path += "/cats" if f[0].isupper() else "/dogs"
		shutil.move(data_path + "/" + f, new_path + "/" + f)

	for f in val_files:
		new_path = val_path
		new_path += "/cats" if f[0].isupper() else "/dogs"
		shutil.move(data_path + "/" + f, new_path + "/" + f)

def create_multiclass_imagefolder():
	train_val_ann_path = "data/annotations/trainval.txt"
	test_ann_path = "data/annotations/test.txt"
	data_path = "data/images/" # Original folder with all images
	class_path = "data/multiclass/"
	subfolders = ["train/", "val/", "test/"]

	if not os.path.isdir(class_path):
		os.mkdir(class_path)

	for sub in subfolders:
		if not os.path.isdir(class_path+sub):
			os.mkdir(class_path+sub)

	train_val_files, train_files, val_files = [], [], []
	test_files = []
	with open(train_val_ann_path) as f:
		for line in f:
			train_val_files.append(line.rstrip())

	with open(test_ann_path) as f:
		for line in f:
			test_files.append(line.rstrip())

	shuffle(train_val_files)
	train_files = train_val_files[:int(train_val_ratio * len(train_val_files))]
	val_files = train_val_files[int(train_val_ratio * len(train_val_files)):]

	for i, l in enumerate([train_files, val_files, test_files]):
		path = class_path + subfolders[i]
		for j in range(37):
			if not os.path.isdir(class_path + subfolders[i] + str(j+1)):
				os.mkdir(class_path + subfolders[i] + str(j+1))
		for f in l:
			img_name, class_id, species, breed = f.split()
			new_path = path + class_id + "/" + img_name + ".jpg"
			img_path = data_path + img_name + ".jpg"
			shutil.move(img_path, new_path)

create_multiclass_imagefolder()
