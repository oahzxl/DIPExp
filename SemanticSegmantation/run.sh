python train_seg_copy.py --batch-size 16 --lr 1e-5 --optim Adam --ckpt-dir ./work_dirs/1 | tee ./work_dirs/1.txt
python train_seg_copy.py --batch-size 16 --lr 6e-5 --optim Adam --ckpt-dir ./work_dirs/2 | tee ./work_dirs/2.txt
python train_seg_copy.py --batch-size 16 --lr 3e-5 --optim Adam --ckpt-dir ./work_dirs/3 | tee ./work_dirs/3.txt
python train_seg_copy.py --batch-size 16 --lr 1e-5 --optim Adam --ckpt-dir ./work_dirs/4 | tee ./work_dirs/4.txt

python train_seg_copy.py --batch-size 16 --lr 1e-2 --optim SGD --ckpt-dir ./work_dirs/5 | tee ./work_dirs/5.txt
python train_seg_copy.py --batch-size 16 --lr 3e-2 --optim SGD --ckpt-dir ./work_dirs/6 | tee ./work_dirs/6.txt
python train_seg_copy.py --batch-size 16 --lr 6e-3 --optim SGD --ckpt-dir ./work_dirs/7 | tee ./work_dirs/7.txt
python train_seg_copy.py --batch-size 16 --lr 2e-2 --optim SGD --ckpt-dir ./work_dirs/8 | tee ./work_dirs/8.txt
