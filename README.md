# DF-CNN
The code is evaluated on Python 3.6 with PyTorch 1.2.0 and MATLAB R2018b

## Step1 Dataset preparation  
Download the datasets you need as below, and extract these datasets to $ROOT_DIR/data/  folder  
wget http://mftp.mmcheng.net/liuyun/rcf/data/bsds_pascal_train_pair.lst  
wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz  
wget http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz  

## Step2 train DF-CNN  
1.Download the datasets you need in step1  
2.Download the model for initialization in [here](https://pan.baidu.com/s/1FLT5fDKmrHneZuAOMzECMA), it's password is 7aqf and then put the model in $ROOT_DIR folder  

## Step3 evaluation
Our best model you can get in [here](https://pan.baidu.com/s/15YrQVeHXFOvocSqXoke5xQ), it's password is yzgz  
1.run the test.py  
2.refer to the evaluation way of [HED](https://github.com/xwjabc/hed)

## Acknowledgment
This code is based on RCF. Thanks to the contributors of RCF.  
>@article{liu2019richer,
>  title={Richer Convolutional Features for Edge Detection},  
>  author={Liu, Yun and Cheng, Ming-Ming and Hu, Xiaowei and Bian, Jia-Wang and Zhang, Le and Bai, Xiang and Tang, Jinhui},  
>  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},  
>  volume={41},  
>  number={8},  
>  pages={1939--1946},  
>  year={2019},  
>  publisher={IEEE}  
>}
The evaluation way has referred to HED. Thanks to the contributors of HED.  
>@inproceedings{xie2015holistically,  
>  title={Holistically-nested edge detection},  
>  author={Xie, Saining and Tu, Zhuowen},  
>  booktitle={Proceedings of the IEEE International Conference on Computer Vision},  
>  pages={1395--1403},  
>  year={2015}  
>}  


