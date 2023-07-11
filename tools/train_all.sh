echo "start evigausbev"
PYTHONPATH=. python tools/train.py --resume --log_dir logs/evigausbev-opv2v/ --cuda_loader
echo "finished evigausbev"

echo "start evibev"
PYTHONPATH=. python tools/train.py --resume --log_dir logs/evibev-opv2v/ --cuda_loader
echo "finished evibev"

echo "start bev"
PYTHONPATH=. python tools/train.py --resume --log_dir logs/bev-opv2v/ --cuda_loader
echo "finished bev"