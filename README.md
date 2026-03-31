# LOPR - shape matching
Matlab implementation for AAAI 2024 paper LOPR:

Yifan Xia, Yifan Lu, Yuan Gao*, and Jiayi Ma*. "Locality Preserving Refinement for Shape Matching with Functional Maps", in Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI), Feb. 2024.

This paper focuses on the pointwise map recovery problem of functional maps in shape-matching. By embedding geometric constraints into the spectral domain, we propose a novel and effective method based on locality consistency, which has proven efficiency and accuracy.

## Dataset 
Relevant shape matching datasets and their corresponding links are available at: https://github.com/XiaYifan1999/Shape-Matching-Dataset-

If you find this project useful, please cite:

```
@inproceedings{xia2024locality,
  title={Locality Preserving Refinement for Shape Matching with Functional Maps},
  author={Yifan Xia, Yifan Lu, Yuan Gao, and Jiayi Ma},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

## Note
Please unzip the gspbox.zip and utils.zip in this directory.

This demo requires Matlab environment and GPU. (GPU accelerates the KNN, and users can convert to CPU.)

## Run the exemplary code
Run the demo.m.

## Acknowledge

The framework implementation is adapted from [MWP](https://github.com/Qinsong-Li/MWP) and [GCPD](https://github.com/AoxiangFan/GeneralizedCoherentPointDrift)
