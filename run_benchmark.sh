export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd MIRAGE
python src/generate.py --config src/config.json
python src/evaluate.py --config src/config.json