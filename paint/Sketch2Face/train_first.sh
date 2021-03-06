python ./train.py \
--name pretrained \
--dataroot ./datasets/data/ \
--no_instance --ngf 48 \
--resize_or_crop scale_width \
--loadSize 256 \
--fineSize 256 \
--batchSize 64 \
--gpu_ids 0 \
--gfm \
--gfm_layer 0,1,2,3 \
--sap_branches 1,5,9,13 \
--niter_fix_global 0 \
--continue_train \
--niter 5 \
--niter_decay 5 