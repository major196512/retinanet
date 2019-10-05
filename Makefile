coco_train:
	CUDA_VISIBLE_DEVICES='0' \
	python train.py \
	--dataset coco \
	--coco_path '/home/taeho/data/MS-COCO/coco2014' \
	--coco_version '2014' \
	--coco_train train \
	--coco_val val \
	--save_name 'coco2014' \
	--depth 50 \
	--lr 1e-5 \
	--batch_size 2 \
	--resume_epoch 14 \
	--trainOnly

tensorboard:
	tensorboard --logdir='./log_coco2014' --port=6006
