## Overview
This project implements an algorithm for 3D reconstruction of satellites using data captured by ground-based amateur telescopes. The algorithm focuses on joint optimization of camera poses and 3D reconstruction. It is based on the framework of [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting).

- **Project Website**: [https://ai4scientificimaging.org/ReconstructingSatellites](https://ai4scientificimaging.org/ReconstructingSatellites)  
- **Paper**: [Reconstructing Satellites in 3D from Amateur Telescope Images](https://arxiv.org/pdf/2404.18394)  
- **Dataset**: [Simulated Data](https://drive.google.com/file/d/1JFwwTmNJD7GqapWC-VUt5xmcyB4yKkuo/view?usp=sharing)  
  - For real-world data, please contact: **[He Sun](https://ai4scientificimaging.org/)**: hesun@pku.edu.cn and **Boyang Liu**: pkulby@foxmail.com  

## Features
- Joint optimization of camera poses and 3D reconstruction.
- Implementation based on the 3D Gaussian Splatting framework.
- Support for both simulated and real-world datasets.

## Installation
The environment setup can follow the instructions from the [3D Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting) or use the provided `requirements.txt` file.

To install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Training
To train the model on the NeRF Synthetic dataset:
```bash
python train.py -s <path to NeRF Synthetic dataset>
```

### Rendering and Evaluation
To generate renderings and compute error metrics:
```bash
python render.py -m <path to trained model>
```

## Dataset
The dataset currently includes simulated data. For real-world telescope data, please contact the authors via the provided email addresses.

## Citation
If you use this project in your research, please cite:
```bibtex
@article{chang2024reconstructing,
  title={Reconstructing satellites in 3d from amateur telescope images},
  author={Chang, Zhiming and Liu, Boyang and Xia, Yifei and Bai, Weimin and Guo, Youming and Shi, Boxin and Sun, He},
  journal={arXiv preprint arXiv:2404.18394},
  year={2024}
}
```

## Acknowledgments
This project is based on the implementation of [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting).