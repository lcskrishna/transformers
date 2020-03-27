export SQUAD_DIR=/data/datasets/wikipedia_bookcorpus_hdf5/download/squad/v1.1/

export HIP_VISIBLE_DEVICES=0

python3.6 examples/run_squad_profile.py \
  --model_type bert \
  --model_name_or_path bert-large-uncased-whole-word-masking \
  --do_train \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir perf \
  --max_steps 10 \
  --overwrite_output_dir
