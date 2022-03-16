# Benchmark4GraphSage

## Basic Config

由于仅比较训练速度，因此infer相关的代码全部删去。

- Dataset: Reddit
- Model: GraphSage, [25, 10]

命令：python xxx.py --batch_size 1024 --epochs 20 --mode uva

## Pyg

```
torch                 1.9.1
torch-cluster         1.5.9
torch-geometric       2.0.3
torch-scatter         2.0.9
torch-sparse          0.6.12
torch-spline-conv     1.2.1
torch-quiver          0.1.1
```

## DGL

```
dgl-cu102             0.8.0

# pip install dgl-cu102 -f https://data.dgl.ai/wheels/repo.html
```

## PGL

由于现在未对PGL层进行设计，所以基本调的是裸的API。Paddle和PGL安装包后续提供，以及对应数据集后续再提供。
1. Paddle whl
2. Pgl whl
