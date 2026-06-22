export PYTHONPATH="/home/ch278262/Desktop/FOMO2JOMO/src:$PYTHONPATH"

python src/extract_features.py \
    --weights /home/ch278262/Desktop/FOMO2JOMO/weights/fomo25_mmunetvae_pretrained.ckpt \
    --data_dir /mnt/8tb/Youwan/ISLES2024/preprocessing_fomo2jomo/isles24/ \
    --out isles24_foundation_features.csv