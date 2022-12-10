# Progressive Modality-shared Transformer (PMT) 
Pytorch code for paper "**Learning Progressive Modality-shared Transformers for Effective Visible-Infrared**

**Person Re-identifification**".

### 1. Results
We adopt the Transformer-based ViT-Base/16 and CNN-based AGW [3] as backbone respectively.

|Datasets    | Backbone | Rank@1  | mAP |  mINP |  Model|
| --------   | -----    | -----  |  -----  | ----- |:----:|
| #SYSU-MM01 | ViT | ~ 67.53% | ~ 64.98% | ~51.86% |[GoogleDrive](https://drive.google.com/file/d/1P6-nI6VRPf1oDYPNlYpVHVz0GH96YmQc/view?usp=share_link)|
|#SYSU-MM01  | AGW* | ~ 67.09% | ~ 64.25% | ~50.89% | [GoogleDrive](https://drive.google.com/file/d/1PjQ9-WEq09ycgQLpSmRa6tYffZ4fdAhQ/view?usp=share_link)|

**\*Both of these two models may have some fluctuation due to random spliting. AGW\* means AGW uses random erasing.  The results might be better by finetuning the hyper-parameters.**

### 2. Datasets

- (1) RegDB [1]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html).

- (2) SYSU-MM01 [2]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

### 3. Requirements

#### **Prepare Pre-trained Model**

- You need to download the ImageNet pretrained transformer model [ViT-Base](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth).

#### Prepare Training Data

- You need to run `python process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.

- You may need mannully define the data path and pre-trained model path in `config.py`.

### 4. Training

**Train PMT by**

```
python train.py --config_file config/SYSU.yml
```
  - `--config_file`:  the config file path.

### 5. Testing

**Test a model on SYSU-MM01 dataset by**

```
python test.py --dataset 'sysu' --mode 'all' --resume 'model_path' --gall_mode 'single' --gpu 0
```
  - `--dataset`: which dataset "sysu" or "regdb".
  - `--mode`: "all" or "indoor"  (only for sysu dataset).
  - `--gall_mode`: "single" or "multi" (only for sysu dataset).
  - `--resume`: the saved model path.
  - `--gpu`: which gpu to use.



**Test a model on RegDB dataset by**

```
python test.py --dataset 'regdb' --resume 'model_path' --trial 1 --tvsearch True --gpu 0
```

  - `--trial`: testing trial which should match the training model  (only for regdb dataset).

  - `--tvsearch`:  whether thermal to visible search  True or False (only for regdb dataset).



### 6. Citation

Most of the code of our backbone are borrowed from [TransReID](https://github.com/damo-cv/TransReID) [4] and [AGW](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) [3]. 

Thanks a lot for the author's contribution.

Please cite the following paper in your publications if it is helpful:

```
@article{lu2022learning,
  title={Learning Progressive Modality-shared Transformers for Effective Visible-Infrared Person Re-identification},
  author={Lu, Hu and Zou, Xuezhang and Zhang, Pingping},
  journal={arXiv preprint arXiv:2212.00226},
  year={2022}
}

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

###  7. References.

[1] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

[2] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

[3] Ye M, Shen J, Lin G, et al. Deep learning for person re-identification: A survey and outlook[J]. IEEE transactions on pattern analysis and machine intelligence, 2021, 44(6): 2872-2893.

[4] He S, Luo H, Wang P, et al. Transreid: Transformer-based object re-identification[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2021: 15013-15022.
