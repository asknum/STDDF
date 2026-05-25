#!/bin/bash

echo "=========================================="
echo "STDDF_MiT Training Script"
echo "=========================================="
python train_mit.py --file_root LEVIR --batch_size 16 --lr 2e-4 --max_epochs 100
