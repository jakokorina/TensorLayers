# Tensor Layers Pytorch Implementation

## Implemented Tensor Networks layers

 - Tensor Train Layer
 - Tensor Ring Layer
 - Tensor Wheel Layer
 - MERA Layer
 - MERA Layer without first layer of disentanglers
 - MERA Layer without second layer of disentanglers
 - MERA Layer without both layers of disentanglers

All of them are successfully working on CIFAR-10 dataset. We do not recommend using rank=1 because it leads to 
sufficient drop in accuracy. Based on our experiments rank=4 is optimal.  

## Installation guide
Needed requirements are written in `requirements.txt`.
```shell
git clone https://github.com/jakokorina/TensorLayers/tree/main
cd TensorLayers
pip install -r requirements.txt
```

## Training on CIFAR-10 dataset
TT Example:
```shell
python3 main.py --layer TT --rank 4 --seed 42 --batch_size 128 --path_to_save "./model.pth" 
```

If you want to use other type of tensor network factorisation, you need to change layer parameter.s
