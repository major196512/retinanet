coco_train:
	CUDA_VISIBLE_DEVICES='0, 1, 2, 3, 4' \
	python train.py \
	--dataset coco \
	--coco_path '/home/ms/data/MS-COCO/coco2014' \
	--coco_version '2014' \
	--coco_train train \
	--coco_val val \
	--save_name 'inverse' \
	--depth 50 \
	--lr 1e-5 \
	--batch_size 2 \
	--trainOnly \
	--inverse

tensorboard:
	tensorboard --logdir='./log_coco2014' --port=6006
