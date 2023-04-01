END=$2
for ((i=1;i<=END;i++)); do
CUDA_VISIBLE_DEVICES=0 wandb agent jimin/uncategorized/$1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 wandb agent jimin/uncategorized/$1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 wandb agent jimin/uncategorized/$1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3 wandb agent jimin/uncategorized/$1 > /dev/null 2>&1 &
done
