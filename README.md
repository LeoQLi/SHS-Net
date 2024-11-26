# SHS-Net: Learning Signed Hyper Surfaces for Oriented Normal Estimation of Point Clouds (TPAMI 2024 / CVPR 2023)

### **[Project](https://leoqli.github.io/SHS-Net/) | [Dataset](https://drive.google.com/drive/folders/1eNpDh5ivE7Ap1HkqCMbRZpVKMQB1TQ6H?usp=share_link) | [arXiv (TPAMI)](https://arxiv.org/abs/2305.05873v2) | [Supplementary (TPAMI)](https://drive.google.com/file/d/1sElnfmWJl4nlq9exfjIkm3J1gdQPranf/view?usp=drive_link) | [arXiv (CVPR)](https://arxiv.org/abs/2305.05873v1) | [Supplementary (CVPR)](https://drive.google.com/file/d/14JMlS62uogc5yooYBzo81zO2eu6ohfd2/view?usp=sharing)**

We propose a novel method called SHS-Net for oriented normal estimation of point clouds by learning signed hyper surfaces, which can accurately predict normals with global consistent orientation from various point clouds. Almost all existing methods estimate oriented normals through a two-stage pipeline, i.e., unoriented normal estimation and normal orientation, and each step is implemented by a separate algorithm. However, previous methods are sensitive to parameter settings, resulting in poor results from point clouds with noise, density variations and complex geometries. In this work, we introduce signed hyper surfaces (SHS), which are parameterized by multi-layer perceptron (MLP) layers, to learn to estimate oriented normals from point clouds in an end-to-end manner. The signed hyper surfaces are implicitly learned in a high-dimensional feature space where the local and global information is aggregated. Specifically, we introduce a patch encoding module and a shape encoding module to encode a 3D point cloud into a local latent code and a global latent code, respectively. Then, an attention-weighted normal prediction module is proposed as a decoder, which takes the local and global latent codes as input to predict oriented normals. Experimental results show that our SHS-Net outperforms the state-of-the-art methods in both unoriented and oriented normal estimation on the widely used benchmarks.

## Requirements

The code is implemented in the following environment settings:
- Ubuntu 16.04
- CUDA 10.1
- Python 3.8
- Pytorch 1.8
- Pytorch3d 0.6
- Numpy 1.23
- Scipy 1.6

## Dataset
We train our network model on the PCPNet dataset.
Our datasets can be downloaded from [here](https://drive.google.com/drive/folders/1eNpDh5ivE7Ap1HkqCMbRZpVKMQB1TQ6H?usp=share_link).
Unzip them to a folder `***/dataset/` and set the value of `dataset_root` in `run.py`.
The dataset is organized as follows:
```
│dataset/
├──PCPNet/
│  ├── list
│      ├── ***.txt
│  ├── ***.xyz
│  ├── ***.normals
│  ├── ***.pidx
├──FamousShape/
│  ├── list
│      ├── ***.txt
│  ├── ***.xyz
│  ├── ***.normals
│  ├── ***.pidx
```

## Train
Our trained model is provided in `./log/001/ckpts/ckpt_800.pt`.
To train a new model on the PCPNet dataset, simply run:
```
python run.py --gpu=0 --mode=train --data_set=PCPNet
```
Your trained model will be save in `./log/***/`.

## Test
You can use the provided model for testing:
- PCPNet dataset
```
python run.py --gpu=0 --mode=test --data_set=PCPNet
```
- FamousShape dataset
```
python run.py --gpu=0 --mode=test --data_set=FamousShape
```
The evaluation results will be saved in `./log/001/results_***/ckpt_800/`.
To test with your trained model, simply run:
```
python run.py --gpu=0 --mode=test --data_set=*** --ckpt_dirs=*** --ckpt_iters=***
```
To save the normals of the input point cloud, you need to change the variables in `run.py`:
```
save_pn = True          # to save the point normals as '.normals' file
sparse_patches = False  # to output sparse point normals or not
```

## Results
Our normal estimation results on the datasets PCPNet and FamousShape can be downloaded from [here](https://drive.google.com/drive/folders/1O606EGHrZaDnlOcH1iQD9GbHEINF2-ox?usp=sharing).

## Citation
If you find our work useful in your research, please cite our paper:

    @inproceedings{li2023shsnet,
      author    = {Li, Qing and Feng, Huifang and Shi, Kanle and Gao, Yue and Fang, Yi and Liu, Yu-Shen and Han, Zhizhong},
      title     = {{SHS-Net}: Learning Signed Hyper Surfaces for Oriented Normal Estimation of Point Clouds},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      publisher = {IEEE Computer Society},
      address   = {Los Alamitos, CA, USA},
      pages     = {13591-13600},
      month     = {June},
      year      = {2023},
      doi       = {10.1109/CVPR52729.2023.01306},
      url       = {https://doi.ieeecomputersociety.org/10.1109/CVPR52729.2023.01306},
    }

    @article{li2024shsnet-pami,
      author    = {Li, Qing and Feng, Huifang and Shi, Kanle and Gao, Yue and Fang, Yi and Liu, Yu-Shen and Han, Zhizhong},
      title     = {Learning Signed Hyper Surfaces for Oriented Point Cloud Normal Estimation},
      booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
      year      = {2024},
      volume    = {46},
      number    = {12},
      pages     = {9957-9974},
      doi       = {10.1109/TPAMI.2024.3431221},
    }
