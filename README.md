# MCD_with_DEV
Implementation of Model Selection for Domain Adpatation 

# Intro

在VisDA2017数据集，Maximum Classifier Discrepancy模型上实现DEV模型选择方法，

DEV中的density ratio通过训练一个输出probability scalar的预判别器 来计算
<img src="https://cdn.mathpix.com/snip/images/1S5h9K6rNdKFVFo0-jksSq4unHdsKVls2F_-KtSDMnA.original.fullsize.png" />

- 使用数据集中的validation split和test split训练域判别器
```bash
bash train_discriminator.sh
```

- 训练出MCD模型，在`main.sh`里调整超参
```bash
bash main.sh
```