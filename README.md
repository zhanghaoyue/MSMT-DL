# MSMT-DL
Source code for Multi-sequence multi-task detection model for Cerebral Venous Thrombus

# Deep Learning Algorithm Enables Cerebral Venous Thrombosis Detection With Routine Brain Magnetic Resonance Imaging
Xiaoxu Yang, [Pengxin Yu](https://github.com/smilenaxx/), [Haoyue Zhang](https://github.com/zhanghaoyue), Rongguo Zhang, Yuehong Liu, Haoyuan Li, Penghui Sun, Xin Liu, Yu Wu, Xiuqin Jia, Jiangang Duan, Xunming Ji, and Qi Yang

[![paper](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_33)

## Abstract
Background:
Cerebral venous thrombosis (CVT) is a rare cerebrovascular disease. Routine brain magnetic resonance imaging is commonly used to diagnose CVT. This study aimed to develop and evaluate a novel deep learning (DL) algorithm for detecting CVT using routine brain magnetic resonance imaging.

Methods:
Routine brain magnetic resonance imaging, including T1-weighted, T2-weighted, and fluid-attenuated inversion recovery images of patients suspected of CVT from April 2014 through December 2019 who were enrolled from a CVT registry, were collected. The images were divided into 2 data sets: a development set and a test set. Different DL algorithms were constructed in the development set using 5-fold cross-validation. Four radiologists with various levels of expertise independently read the images and performed diagnosis within the test set. The diagnostic performance on per-patient and per-segment diagnosis levels of the DL algorithms and radiologist’s assessment were evaluated and compared.

Results:
A total of 392 patients, including 294 patients with CVT (37±14 years, 151 women) and 98 patients without CVT (42±15 years, 65 women), were enrolled. Of these, 100 patients (50 CVT and 50 non-CVT) were randomly assigned to the test set, and the other 292 patients comprised the development set. In the test set, the optimal DL algorithm (multisequence multitask deep learning algorithm) achieved an area under the curve of 0.96, with a sensitivity of 96% (48/50) and a specificity of 88% (44/50) on per-patient diagnosis level, as well as a sensitivity of 88% (129/146) and a specificity of 80% (521/654) on per-segment diagnosis level. Compared with 4 radiologists, multisequence multitask deep learning algorithm showed higher sensitivity both on per-patient (all P<0.05) and per-segment diagnosis levels (all P<0.001).

Conclusions:
The CVT-detected DL algorithm herein improved diagnostic performance of routine brain magnetic resonance imaging, with high sensitivity and specificity, which provides a promising approach for detecting CVT.


Framework illustration:
![image](https://user-images.githubusercontent.com/11541770/199294166-5c316fe5-7af0-4bd0-bc9e-242648fd29f4.png)

For MRI preprocessing pipeline, please refer to this repository from our previous work:\
https://github.com/zhanghaoyue/stroke_preprocessing

## Citation
If you find our work useful, please consider citing:
```
@article{yang2023deep,
  title={Deep Learning Algorithm Enables Cerebral Venous Thrombosis Detection With Routine Brain Magnetic Resonance Imaging},
  author={Yang, Xiaoxu and Yu, Pengxin and Zhang, Haoyue and Zhang, Rongguo and Liu, Yuehong and Li, Haoyuan and Sun, Penghui and Liu, Xin and Wu, Yu and Jia, Xiuqin and others},
  journal={Stroke},
  year={2023},
  publisher={Am Heart Assoc}
}
```
