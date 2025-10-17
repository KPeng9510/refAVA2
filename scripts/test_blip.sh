# ViT-B/16 # pami_method_retry_30_08_try6 # pami_method_retry_31_08_try6_v2
job_name="pami_ablation_agg_spatial"
 python -m torch.distributed.launch --nproc_per_node=4 \
    train_pami.py --do_train --num_thread_reader=4\
    --epochs=40 --batch_size=48 --n_display=25 \
    --output_dir ckpts3/${job_name} \
    --lr 1e-4 --max_words 32 --max_frames 8 --batch_size_val 32 \
    --datatype ravar \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0 --slice_framepos 2 --cross_model cross-base --loose_type \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/16 2>&1 | tee -a log/${job_name}
