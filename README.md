## Overview
This project implements an algorithm for 3D reconstruction of satellites using data captured by ground-based amateur telescopes. The algorithm focuses on joint optimization of camera poses and 3D reconstruction. It is based on the framework of [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting).

- **Project Website**: [https://ai4scientificimaging.org/ReconstructingSatellites](https://ai4scientificimaging.org/ReconstructingSatellites)  
- **Paper**: [Reconstructing Satellites in 3D from Amateur Telescope Images](https://arxiv.org/pdf/2404.18394)  
- **Dataset**: [Simulated Data](https://drive.google.com/file/d/1JFwwTmNJD7GqapWC-VUt5xmcyB4yKkuo/view?usp=sharing)  
  - For real-world data, please contact: **[He Sun](https://ai4scientificimaging.org/)**: hesun@pku.edu.cn and **Boyang Liu**: pkulby@foxmail.com  

## Features
- Joint optimization of camera poses and 3D reconstruction.
- Preceding **pose estimation pipeline** with:
  1. **Feature Extraction & Matching** ([Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization))  
  2. **Orthogonal Model Initialization** ([OrthoPose](https://www.ipol.im/pub/art/2019/248/?utm_source=doi))  
  3. **Incremental SfM Reconstruction**  
- Implementation based on the 3D Gaussian Splatting framework.
- Support for both simulated and real-world datasets.

## Data Preprocessing

Before pose estimation and 3D reconstruction, the raw telescope images and videos undergo a **data preprocessing pipeline** to improve quality and robustness. This step significantly reduces noise, corrects distortions, and enhances fine details.

### 1. Image Stacking (AutoStakkert)
- Tool: [AutoStakkert](https://www.astrokraai.nl/software/latest.php)  
- Purpose: Align and stack multiple frames of the same satellite image.  
- Benefit: Effectively reduces random noise and mitigates atmospheric turbulence effects, resulting in a sharper and more stable image.

### 2. Image Enhancement (RegiStax)
- Tool: [RegiStax](https://www.astronomie.be/registax/)  
- Purpose: Apply wavelet-based sharpening and contrast enhancement.  
- Benefit: Highlights structural details of the satellite and improves feature visibility for the downstream pose estimation module.

### 3. Video Denoising (VRT)
- Tool: [VRT - Video Restoration Transformer](https://github.com/JingyunLiang/VRT)  
- Purpose: Perform deep-learning-based video denoising and restoration.  
- Benefit: Produces cleaner video sequences with reduced noise, which helps in maintaining temporal consistency across frames.

---

This preprocessing pipeline ensures that the data fed into the **pose estimation** and **3D reconstruction** stages is of the highest possible quality, improving the reliability of both feature extraction and final 3D reconstruction.


## Pose Estimation Pipeline

Before running the Gaussian Splatting stage, camera poses are estimated through a **three-step pipeline**:

### 1. Feature Point Extraction and Matching (Hierarchical-Localization)
- Install environment following the [Hierarchical-Localization repo](https://github.com/cvg/Hierarchical-Localization).  
- Modify the image directory in `Hierarchical-Localization/test.py`:
  ```python
  images = Path('datasets/simu1')
  ```
- Modify the image size in `Hierarchical-Localization/hloc/extract_features.py`:
  ```        
  # Change zhe resize_max as the size of images, for example: css:360 iss:640
  "preprocessing": {
      "grayscale": True,
      "resize_max": 360,
  },
  ```
- Run feature extraction and matching:
  ```bash
  python test.py
  ```
- Outputs include:
  - `01.txt`, `02.txt`, `12.txt` (pairwise matches)  
  - `features.pkl`, `keypoints.pkl`, `matches.pkl`  
- Copy these files into the **OrthoPose** module folder.

---

### 2. Orthogonal Model Initialization (OrthoPose)
- Follow the [OrthoPose repository](https://www.ipol.im/pub/art/2019/248/?utm_source=doi) for environment setup.  
- Modify dataset paths in `example.m` and `mainPoseEstimation.m` (e.g., replace `simu2/` with the directory from step 1).  
- Run:
  ```matlab
  example.m
  ```
- Outputs include:
  - `corr.csv` (correspondences)  
  - `camera.csv` (initial camera parameters)  
  - `points.csv` (initial 3D points)  
- Copy these files (all *.pkl and all *.csv) into the **SfM** module folder.

---

### 3. Incremental Reconstruction (SfM)
- Using the outputs above, select the corresponding Python script (`sfm_css`, `sfm_iss`, or `sfm_simu1-3`).  
- Each script has minimal configuration differences. Run the chosen script to obtain, which could be visivisualization by function visualization():
  - `optimized_points_3d.npy`  
  - `optimized_camera.npy`  
- These represent the refined camera poses and sparse 3D points.  
- Finally, get **NeRF-compatible dataset** (`optimized_points_3d.npy`, `transforms_train.json`,`transforms_test.json`,`transforms_val.json` ) for Gaussian Splatting.


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