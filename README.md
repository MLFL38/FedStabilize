# FedStabilize
Code for **FedStabilize**, a two-stage federated learning framework for robust training under label noise and data heterogeneity. 
This codebase was initially adapted from the official implementation of [FedCorr](https://github.com/Xu-Jingyi/FedCorr), which has been restructured to implement the two-stage FedStabilize framework, including prototype-based feature alignment and confidence-aware sample filtering.

---

## Training Examples

##### CIFAR-10 (Non-IID: `{p=0.3, α_dir=10}`, Noise level: `{ρ=0.6, τ=0.5}`)
```bash
python3.10 main.py --dataset cifar10 --model resnet18 --non_iid_prob_class 0.3 --alpha_dirichlet 10 --level_n_system 0.6 --level_n_lowerb 0.5 --rounds1 40 --rounds2 950 --seed 1 --mixup --lr 0.03 --confidence_thres 0.6
```

##### CIFAR-10 (IID, Noise level: `{ρ=0.6, τ=0.5}`)
```bash
python3.10 main.py --dataset cifar10 --model resnet18 --iid --level_n_system 0.6 --level_n_lowerb 0.5 --rounds1 40 --rounds2 950 --seed 1 --mixup --lr 0.03  --confidence_thres 0.6
```

##### CIFAR-100 (Non-IID: `{p=0.3, α_dir=10}`, Noise level: `{ρ=0.4, τ=0}`)
```bash
python3.10 main.py --dataset cifar100 --model resnet34 --num_users 50 --non_iid_prob_class 0.3 --alpha_dirichlet 10 --level_n_system 0.4 --level_n_lowerb 0 --rounds1 90  --rounds2 900 --seed 1 --mixup --lr 0.01 --confidence_thres 0.5
```

##### Clothing1M (Non-IID: `{p=0.7, α_dir=10}`)
```bash
python3.10 main_clothing.py --dataset clothing1m --model resnet50 --pretrained --num_users 500 --non_iid_prob_class 0.7 --alpha_dirichlet 10 --frac 0.02 --level_n_system 0 --level_n_lowerb 0 --rounds1 50 --rounds2 100 --local_bs 16 --lr 0.001 --seed 1 --mixup --confidence_thres 0.6
```
>  **Note for Clothing1M**  
> In this experiment, noisy clients **use all their local data** and **apply confidence-based relabeling only to confidently predicted samples**. 
> Clothing1M contains real-world, instance-dependent noise. In such cases, even mislabeled examples can still be useful for generalization. Therefore, we retain all samples and avoid overcorrecting the data. Only high-confidence samples are relabeled to prevent introducing additional noise or bias from uncertain predictions.







