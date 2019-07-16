# Knowledge Distillation
Teacher-Student Mechanism to compress model & improve compact models
 
###Related Works
* [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531v1)
* [Paying More Attention to Attention: Improving the Performance of 
Convolutional Neural Networks via Attention Transfer](https://arxiv.org/abs/1612.03928v3)
* [Like What You Like: Knowledge Distill via Neuron Selectivity Transfer](https://arxiv.org/abs/1707.01219)
* [A Gift From Knowledge Distillation: Fast Optimization, 
Network Minimization and Transfer Learning](https://zpascal.net/cvpr2017/Yim_A_Gift_From_CVPR_2017_paper.pdf)
* [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550)

###Implementation Details
#####1. Distilling the Knowledge in a Neural Network
Known as KD; Use soft label produced by teacher model to direct student model