# # CUDA_VISIBLE_DEVICES=4 python generate_sam2_masks.py \
# #   --input_dir /workspace/semantic_gs/gaussian-splatting/data/db/drjohnson/images \
# #   --output_dir /workspace/semantic_gs/gaussian-splatting/data/db/drjohnson/semantic_gt \
# #   --sam2_config /workspace/semantic_gs/sam2/sam2/configs/sam2/sam2_hiera_s.yaml \
# #   --sam2_checkpoint /workspace/semantic_gs/sam2/checkpoints/sam2.1_hiera_small.pt \
# #   --device cuda

  

# CUDA_VISIBLE_DEVICES=4 python generate_sam2_masks.py \
#   --input_dir /workspace/semantic_gs/gaussian-splatting/data/db/drjohnson/images \
#   --output_dir /workspace/semantic_gs/gaussian-splatting/data/db/drjohnson/semantic_gt \
#   --sam2_config configs/sam2.1/sam2.1_hiera_s.yaml \
#   --sam2_checkpoint /workspace/semantic_gs/sam2/checkpoints/sam2.1_hiera_small.pt \
#   --device cuda
  
python generate_sam2_masks.py \
  --input_dir /workspace/semantic_gs/gaussian-splatting/data/db/drjohnson/images \
  --output_dir /workspace/semantic_gs/gaussian-splatting/data/db/drjohnson/semantic_gt \
  --sam2_config configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2_checkpoint /workspace/semantic_gs/sam2/checkpoints/sam2.1_hiera_small.pt \
  --device cpu \
  --points_per_side 8


  
python /workspace/semantic_gs/gaussian-splatting/scripts/generate_sam2_masks.py \
  --input_dir /workspace/semantic_gs/gaussian-splatting/data/tandt/train/images \
  --output_dir /workspace/semantic_gs/gaussian-splatting/data/tandt/train/semantic_gt \
  --sam2_config configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2_checkpoint /workspace/semantic_gs/sam2/checkpoints/sam2.1_hiera_small.pt \
  --device cpu \
  --points_per_side 8


python /workspace/semantic_gs/gaussian-splatting/scripts/generate_sam2_masks.py \
  --input_dir /workspace/semantic_gs/gaussian-splatting/data/tandt/truck/images \
  --output_dir /workspace/semantic_gs/gaussian-splatting/data/tandt/truck/semantic_gt \
  --sam2_config configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2_checkpoint /workspace/semantic_gs/sam2/checkpoints/sam2.1_hiera_small.pt \
  --device cpu \
  --points_per_side 8
