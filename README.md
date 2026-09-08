# DCP-Det
**DCP-Det: A Defect Contrast Prior-Guided Detector for X-ray Weld Defect Detection**
This repository provides the official implementation of **DCP-Det**, a defect contrast prior-guided object detector developed for X-ray weld defect detection.
DCP-Det explicitly incorporates local radiographic contrast information into both **multi-scale feature representation** and **bounding-box localization optimization**. In addition, this repository provides the official data split and annotation information for the **Engineering Nondestructive-testing Dataset (END)**.
## Overview of the Architecture
Automatic defect detection in X-ray weld radiographs is challenging because defects often exhibit:
- weak local contrast;
- irregular morphology;
- large scale variation;
- complex radiographic backgrounds;
- multiple defect instances within a single image;
- multiple defect categories within the same radiograph.
To address these challenges, we propose **DCP-Det**, which introduces a parameter-free **Defect Contrast Prior (DCP)** derived from local radiographic intensity deviations.
<img width="2324" height="1029" alt="image" src="https://github.com/user-attachments/assets/1fd71247-7a0e-4a83-94f3-a1e4a83f2504" />
The DCP is incorporated into the detector at two complementary levels:
1. **DCP-EVC**  
   Integrates defect-sensitive contrast information with context-aware feature enhancement at high-level pyramid features.
2. **DCP-DIoU**  
   Uses DCP-derived contrast characteristics to adaptively weight positive RoI regression losses, emphasizing contrast-difficult samples during bounding-box refinement.
## END Dataset
The **Engineering Nondestructive-testing Dataset (END)** contains:
- **1,180 X-ray radiographs**
- **2,957 annotated defect instances**
- **5 defect categories**
The five categories are:
| ID | Category |
|---:|---|
| 1 | Crack |
| 2 | LF |
| 3 | IP |
| 4 | SI |
| 5 | Pore |
Unlike simplified settings in which radiographs are decomposed into isolated single-defect samples, END preserves the original radiographic scenes. Therefore, multiple defect instances and different defect categories may coexist within the same image.
The dataset from Xiangyang Central Hospital can be downloaded using the following links:Google Drive:[(https://drive.google.com/uc?export=download&id=11CiATDjSktr5mObumnQQ6DIikDazKfj2)] Please verify the numerical category IDs against the released annotation files before using them in a custom implementation.
## Installation
The reference environment uses Python 3.8.18, PyTorch 2.1.0, torchvision 0.16.0, and a CUDA 12.1 PyTorch build. Install a PyTorch build appropriate foryour platform, then install the remaining dependencies.
