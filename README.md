# FedStabilize
Code for FedStabilize

To train on CIFAR-10 with the non-IID setting {p=0.3, alpha_dir=10}:

python3.10 main.py --dataset cifar10 --model resnet18 --non_iid_prob_class 0.3 --alpha_dirichlet 10 --level_n_system 0.6 --level_n_lowerb 0.5 --rounds1 40 --rounds2 950 --seed 1 --mixup --lr 0.03  --confidence_thres 0.6

To train on CIFAR-10 with IID setting :
python3.10 main.py --dataset cifar10 --model resnet18 --iid --level_n_system 0.6 --level_n_lowerb 0.5 --rounds1 40 --rounds2 950 --seed 1 --mixup --lr 0.03  --confidence_thres 0.6

To train on CIFAR-100 with the non-IID  setting {p=0.3, alpha_dir=10}:
python3.10 main.py --dataset cifar100 --model resnet34 --num_users 50 --non_iid_prob_class 0.3 --alpha_dirichlet 10 --level_n_system 0.4 --level_n_lowerb 0 --rounds1 90  --rounds2 900 --seed 1 --mixup --lr 0.01 --confidence_thres 0.5

To train on Clothing1M with the non-IID setting {p=0.7, alpha_dir=10}:
python3.10 main_clothing.py --dataset clothing1m --model resnet50 --pretrained --num_users 500 --non_iid_prob_class 0.7 --alpha_dirichlet 10 --frac 0.02 --level_n_system 0 --level_n_lowerb 0 --rounds1 50 --rounds2 100 --local_bs 16 --lr 0.001 --seed 1 --mixup --confidence_thres 0.6


