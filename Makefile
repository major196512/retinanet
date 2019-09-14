coco_train:
	CUDA_VISIBLE_DEVICES='0' \
	python train.py \
	--dataset coco \
	--coco_path '/home/taeho/data/MS-COCO' \
	--coco_class '2014' \
	--coco_train train \
	--coco_val val \
	--save_name 'coco2014' \
	--log_dir 'log_coco' \
	--resume_epoch 0 \
	--lr 1e-5 \
	--batch_size 2 \
	--depth 50

tensorboard:
	tensorboard --logdir='./log_coco' --port=6007
