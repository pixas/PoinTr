srun -p NLP --gres=gpu:4  -N1 bash ./scripts/dist_train.sh 4 20471 \
--config ./cfgs/PCN_models/PoinTr.yaml \
--save_dir ~/checkpoints/pointr_experiments \
--exp_name mha_cross_debug_in1536_out3584 

srun -p NLP --gres=gpu:8 -N1 --quotatype=auto bash ./scripts/dist_train.sh 8 20263 \
--config ./cfgs/PCN_models/PoinTr_amlp.yaml \
--save_dir /mnt/petrelfs/jiangshuyang/checkpoints/pointr_experiments \
--exp_name amlp_self2_in1536_out3584 

srun -p NLP --gres=gpu:4 -N1 --quotatype=auto bash ./scripts/dist_train.sh 4 19999 \
--config ./cfgs/PCN_models/PoinTr_mempool.yaml \
--save_dir /mnt/petrelfs/jiangshuyang/checkpoints/pointr_experiments \
--exp_name mempool_in1536_out3584

srun -p NLP --gres=gpu:1 --quotatype=reserved -N1 bash ./scripts/test.sh 0 \
--save_dir /mnt/cache/jiangshuyang/checkpoints/pointr_experiments \
--exp_name amlp_cross_in1536_out3584 \
--config ./cfgs/PCN_models/PoinTr_amlpseq.yaml \
--ckpts /mnt/petrelfs/jiangshuyang/checkpoints/pointr_experiments/PoinTr_amlpseq/PCN_models/amlp_cross_in1536_out3584/ckpt-best.pth

srun -p NLP --gres=gpu:1  -N1 bash ./scripts/test.sh 0 \
--save_dir ~/checkpoints/pointr_experiments \
--exp_name amlp_self_in1536_out3584 \
--config ./cfgs/PCN_models/PoinTr_amlp.yaml \
--ckpts ~/checkpoints/pointr_experiments/PoinTr_amlp/PCN_models/amlp_self_in1536_out3584/ckpt-best.pth
