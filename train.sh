MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun \
         --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=ImageNet --partition=RL_DSK_Face \
        ./scripts/imagenet_micro_search.sh
