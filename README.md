#  Attention-based Multiple Instance Mutation Learning (AMIML) model
This is the introduction of our proposed Mutiple Instance Learning(MIL) algorithm.

* [Our pipeline of data proprecessing](Pipeline)
* [Our clustering algorithm](前端相关书籍)
* [Proposed MIL algorithm](cc相关书籍)
* [Visualization](cc相关书籍)


## Our pipeline of data proprecessing
We preprecess the whole-slide images using a pipeline in our laboratory and extract the feature vectors of the pathology images
![image](https://github.com/Boomwwe/AMIML/blob/main/Figure/Figure_1(a).png)
And more details and the code of this pipeline can be seen in [https://github.com/1996lixingyu1996/CRCNet](https://github.com/1996lixingyu1996/CRCNet)
## Our clustering algorithm

## Our clustering algorithm
We use the unsupervised clustering method to select predictive image patches
![image](https://github.com/Boomwwe/AMIML/blob/main/Figure/Figure_1(b).png)


## Proposed MIL algorithm
Our MIL algorithm is an adaptation of the CHOWDER algorithm. AMIML is comprised of four important components: (1) an encoder, (2) a decoder, (3) a residual
weight connection, and (4) an Attention mechanism.
![image](https://github.com/Boomwwe/AMIML/blob/main/Figure/Figure_1(c).png)

## Visualization
The attention mechanism is designed to assign higher weights to the more relevant patches to the outcome. This feature allows us to identify predictive image patches.
![image](https://github.com/Boomwwe/AMIML/blob/main/Figure/Figure_4.png)

