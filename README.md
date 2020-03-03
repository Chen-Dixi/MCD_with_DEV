# MCD_with_DEV
Implementation of Deep Embedded Validation for Domain Adpatation on [visda2017](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) dataset with [MCD](https://github.com/mil-tokyo/MCD_DA) model

# Required
- python3.6
- pytorch 1.2.0
- [dixitool](github.com/chen-dixi/dixitool)
- tqdm
- numpy

# Intro

在VisDA2017数据集，Maximum Classifier Discrepancy模型上实现DEV模型选择方法，

DEV中的density ratio通过训练一个输出probability scalar的预判别器 来计算
<div align=center>
    <img src="https://cdn.mathpix.com/snip/images/1S5h9K6rNdKFVFo0-jksSq4unHdsKVls2F_-KtSDMnA.original.fullsize.png" />
</div>

只是在深度DomainAdaptation模型上 对DEV进行粗略的实现，计算论文中的无偏估计
- 训练出MCD模型，在`main.sh`里调整超参，把数据集的路径修改为自己的
```bash
bash main.sh
```

- 使用数据集中的validation split和test split训练域判别器，判别器的模型在MCD/models.py中，参考SinGAN中的域判别器，用卷积对3x224x224的图片进行下采样。
```bash
bash train_discriminator.sh
```

- 用前两步训练的模型，在数据集的validation split上面计算相对于test split的无偏估计
```bash
bash DEV.sh
```


