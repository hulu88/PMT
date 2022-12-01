# Progressive Modality-shared Transformer (PMT) 
Pytorch code for paper "**Learning Progressive Modality-shared Transformers for Effective Visible-Infrared**

**Person Re-identifification**".

We adopt the ViT-Base/16 as the backbone. We also apply our proposed method on CNN-based backbone AGW [3].

|Datasets    | Backbone | Rank@1  | mAP |  mINP |  Model|
| --------   | -----    | -----  |  -----  | ----- |:----:|
| #SYSU-MM01 | ViT | ~ 67.53% | ~ 64.98% | ~51.86% |[GoogleDrive](https://drive.google.com/file/d/1PjQ9-WEq09ycgQLpSmRa6tYffZ4fdAhQ/view?usp=share_link)|
|#SYSU-MM01  | AGW* | ~ 67.09% | ~ 64.25% | ~50.89% | [GoogleDrive](https://drive.google.com/file/d/1PjQ9-WEq09ycgQLpSmRa6tYffZ4fdAhQ/view?usp=share_link)|

*Both of these two datasets may have some fluctuation due to random spliting. AGW***** means AGW method with random erasing.  The results might be better by finetuning the hyper-parameters. 

### 1. Datasets.

- (1) RegDB [1]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html).

- (2) SYSU-MM01 [2]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

   

### 2. Citation

Most of the code of our backbone are borrowed from [TransReID](https://github.com/damo-cv/TransReID) [4] and [AGW](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) [3]. 

Thanks a lot for the author's contribution.

Please cite the following paper in your publications if it is helpful:

```
@inproceedings{he2021transreid,
  title={Transreid: Transformer-based object re-identification},
  author={He, Shuting and Luo, Hao and Wang, Pichao and Wang, Fan and Li, Hao and Jiang, Wei},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={15013--15022},
  year={2021}
}

@article{ye2021deep,
  title={Deep learning for person re-identification: A survey and outlook},
  author={Ye, Mang and Shen, Jianbing and Lin, Gaojie and Xiang, Tao and Shao, Ling and Hoi, Steven CH},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={44},
  number={6},
  pages={2872--2893},
  year={2021},
  publisher={IEEE}
}
```

###  3. References.

[1] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

[2] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

[3] Ye M, Shen J, Lin G, et al. Deep learning for person re-identification: A survey and outlook[J]. IEEE transactions on pattern analysis and machine intelligence, 2021, 44(6): 2872-2893.

[4] He S, Luo H, Wang P, et al. Transreid: Transformer-based object re-identification[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2021: 15013-15022.
