python main.py --optim Adam --box 10 --lr 3e-4 --head base | tee ./log/1.txt
python main.py --optim Adam --box 10 --lr 1e-4 --head base | tee ./log/2.txt
python main.py --optim Adam --box 10 --lr 1e-3 --head base | tee ./log/3.txt

python main.py --optim Adam --box 5 --lr 3e-4 --head base | tee ./log/4.txt
python main.py --optim Adam --box 5 --lr 1e-4 --head base | tee ./log/5.txt
python main.py --optim Adam --box 5 --lr 1e-3 --head base | tee ./log/6.txt

python main.py --optim Adam --box 1 --lr 3e-4 --head base | tee ./log/7.txt
python main.py --optim Adam --box 1 --lr 1e-4 --head base | tee ./log/8.txt
python main.py --optim Adam --box 1 --lr 1e-3 --head base | tee ./log/9.txt

python main.py --optim SGD --box 10 --lr 1e-3 --head base | tee ./log/10.txt
python main.py --optim SGD --box 10 --lr 3e-3 --head base | tee ./log/11.txt
python main.py --optim SGD --box 10 --lr 1e-2 --head base | tee ./log/12.txt

python main.py --optim SGD --box 5 --lr 1e-3 --head base | tee ./log/13.txt
python main.py --optim SGD --box 5 --lr 3e-3 --head base | tee ./log/14.txt
python main.py --optim SGD --box 5 --lr 1e-2 --head base | tee ./log/15.txt

python main.py --optim SGD --box 1 --lr 1e-3 --head base | tee ./log/16.txt
python main.py --optim SGD --box 1 --lr 3e-3 --head base | tee ./log/17.txt
python main.py --optim SGD --box 1 --lr 1e-2 --head base | tee ./log/18.txt

python main.py --optim Adam --box 10 --lr 3e-4 --head fpn | tee ./log/19.txt
python main.py --optim Adam --box 10 --lr 1e-4 --head fpn | tee ./log/20.txt
python main.py --optim Adam --box 10 --lr 1e-3 --head fpn | tee ./log/21.txt

python main.py --optim Adam --box 5 --lr 3e-4 --head fpn | tee ./log/22.txt
python main.py --optim Adam --box 5 --lr 1e-4 --head fpn | tee ./log/23.txt
python main.py --optim Adam --box 5 --lr 1e-3 --head fpn | tee ./log/24.txt

python main.py --optim Adam --box 1 --lr 3e-4 --head fpn | tee ./log/25.txt
python main.py --optim Adam --box 1 --lr 1e-4 --head fpn | tee ./log/26.txt
python main.py --optim Adam --box 1 --lr 1e-3 --head fpn | tee ./log/27.txt

python main.py --optim SGD --box 10 --lr 1e-3 --head fpn | tee ./log/28.txt
python main.py --optim SGD --box 10 --lr 3e-3 --head fpn | tee ./log/29.txt
python main.py --optim SGD --box 10 --lr 1e-2 --head fpn | tee ./log/30.txt

python main.py --optim SGD --box 5 --lr 1e-3 --head fpn | tee ./log/31.txt
python main.py --optim SGD --box 5 --lr 3e-3 --head fpn | tee ./log/32.txt
python main.py --optim SGD --box 5 --lr 1e-2 --head fpn | tee ./log/33.txt

python main.py --optim SGD --box 1 --lr 1e-3 --head fpn | tee ./log/34.txt
python main.py --optim SGD --box 1 --lr 3e-3 --head fpn | tee ./log/35.txt
python main.py --optim SGD --box 1 --lr 1e-2 --head fpn | tee ./log/36.txt
