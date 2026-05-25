#!/bin/bash

echo "=========================================="
echo "STDDF_MiT Testing Script"
echo "=========================================="

python test_mit.py --file_root LEVIR --lr 2e-4 --max_epochs 100
