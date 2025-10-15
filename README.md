# HMC-Net


## ✨ Framework

<center>
  <img src="https://github.com/jylEcho/HMC-Net/blob/main/images/V11.png" width="800" alt="">
</center>


## Overview

This repository provides the official implementation of CSF-Net++, a clinical-guided spatiotemporal dual-configuration network designed for accurate liver tumor segmentation from contrast-enhanced CT (CECT).
Building upon prior multi-phase fusion strategies, CSF-Net++ introduces two upgraded configurations — a spatial clinical-guided fusion module and a temporal dual-configuration mechanism — to more effectively capture both inter-phase dependencies and intra-phase feature dynamics.

Unlike existing methods that treat all phases equally, CSF-Net++ leverages the clinical dominance of the portal venous (PV) phase and explicitly models phase-specific propagation orders, enabling more reliable multi-phase feature integration.

Extensive experiments on PLC-CECT and MPLL datasets demonstrate that CSF-Net++ consistently outperforms previous approaches (e.g., MW-UNet, MCDA-Net), establishing a new state-of-the-art in both segmentation accuracy and boundary precision.

## Datasets  

| Dataset                                          | Phases               | Disease types  |
| ------------------------------------------------ | -------------------- | -------------- |
| [PLC-CECT](https://github.com/ljwa2323/PLC_CECT) | Multi (NC/ART/PV/DL) | 152,965 slices |
| MPLL                                             | Multi (ART/PV/DL)    | 952,601 slices |

<img src="https://github.com/jylEcho/ICLR26_CSF-Net/blob/main/img/Swin-V5.0.png" width="500">

## 👉 Why Multi-Phase CECT?

Contrast-enhanced CT captures dynamic enhancement patterns via multiple phases:  

- **NC (non-contrast):** baseline anatomy  
- **ART (arterial):** highlights early vascular supply  
- **PV (portal venous):** best lesion–parenchyma contrast  
- **DL (delayed):** reveals washout and boundary refinement 

These phases are complementary, making multi-phase fusion a powerful strategy for robust lesion segmentation.

## Limitations of Existing Fusion Methods

- **Input-level fusion:** simple concatenation, ignores phase importance  
- **Feature-level fusion:** self-attention, but equal weighting across phases  
- **Decision-level fusion:** ensemble of outputs, but lacks inter-phase guidance  

➡️ A key drawback: they treat all phases equally, ignoring **clinical hierarchy (PV > ART > DL)**.  

## Our Contributions

- **Systematic single-phase analysis**: On the MPLL dataset, the portal venous (PV) phase demonstrates the strongest segmentation performance, which aligns with its established clinical dominance in liver tumor assessment.
- **Clinically-guided propagation order**: CSF-Net++ introduces a Multi-Phase Cross-Query Sequential (MCQS) mechanism that explicitly models clinically consistent propagation. Ablation results (Table 3) confirm that the PV→ART→DL order achieves the best performance (76.29% DSC, 61.67% Jaccard), with the lowest HD$_{95}$ and ASSD. 
- **Proposed network:** The upgraded CSF-Net++ incorporates two enhanced configurations — a spatial clinical-guided fusion module and a temporal dual-configuration strategy — enabling more effective cross-phase feature interaction and fine-grained refinement for robust multi-phase liver tumor segmentation. 


## Results

- **On PLC-CECT:** Both configurations of CSF-Net++ achieve superior results over existing methods.
CSF-Swin attains 76.26% DSC (+1.09%), 62.44% Jaccard (+2.22%), 24.63 HD$_{95}$ (↓4.59), and 14.67 ASSD (↓2.08), surpassing MCDA-Net and MW-UNet by a clear margin.
CSF-CNN also delivers strong performance (76.14% DSC, 61.47% Jaccard), confirming the robustness of the proposed dual-configuration framework across backbone designs.

- **On MPLL:** CSF-Net++ establishes new state-of-the-art results.
CSF-Swin achieves 76.29% DSC and 61.67% Jaccard, with the lowest HD$_{95}$ and ASSD, while
CSF-CNN attains 75.96% DSC and 61.23% Jaccard, demonstrating consistent cross-dataset generalization and clinical reliability.

- **Qualitative analysis:** Compared to MCDA-Net, CSF-Net++ produces sharper tumor boundaries (e.g., case 1) and more accurate identification of small nodules (e.g., case 3), validating the effectiveness of its spatiotemporal dual-configuration design for multi-phase feature fusion.

Extensive experiments on two benchmark datasets, LiTS2017 and MPLL, demonstrate the superiority of our proposed method, which significantly outperforms existing state-of-the-art approaches.



🚀 ## Quick Start Guide
Environment Setup

To ensure reproducibility, we recommend using Conda for a clean Python environment.

Operating System: Linux 5.4.0

Create the environment:

conda create -n liverseg_env python=3.8 -y
conda activate liverseg_env


Install PyTorch (with CUDA 11.7 support):

pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117


Install dependencies:

pip install numpy==1.14.2 pandas==0.23.3 scipy==1.0.0 tqdm==4.40.2
pip install scikit-image==0.13.1 SimpleITK==1.0.1 pydensecrf==1.0rc3 visdom==0.1.8.8

🏋️‍♂️ Data Processing and Training Pipeline

This project focuses on multi-phase liver tumor segmentation. The workflow is as follows:

1️⃣ Dataset Split

Divide patients into training, validation, and testing subsets:

python multi_phase/multi_phase/dataset_prepare/generate_patients_txt.py

2️⃣ Liver Bounding Box Generation

Generate liver bounding boxes to improve training efficiency and reduce background noise:

python multi_phase/multi_phase/dataset_prepare/generate_liverbox.py


📦 Output: liver ROI annotations.

3️⃣ Data Preprocessing

Convert 3D volumes into 2D slices for model training and evaluation:

Generate testing slices:

python multi_phase/multi_phase/dataset_prepare/rawdata_2D_test.py


Generate training slices:

python multi_phase/multi_phase/dataset_prepare/rawdata_2D_train.py


🖼️ Output folders: process_data/train/ and process_data/test/

4️⃣ Generate Training/Test Index Files

Create .txt files listing the dataset for easy loading:

python multi_phase/multi_phase/dataset_prepare/get_txt.py


📂 Example outputs:

multi_phase/multi_phase/lists/lists_liver/train.txt

multi_phase/multi_phase/lists/lists_liver/test_vol.txt

5️⃣ Training

Launch training using the provided script:

export CUDA_VISIBLE_DEVICES=0,1,2
cd ..
python train.py \
  --n_gpu 3 \
  --root_path /path/to/train/ \
  --test_path /path/to/test/ \
  --module /path/to/model/ \
  --dataset Multiphase \
  --eval_interval 5 \
  --max_epochs 100 \
  --batch_size 8 \
  --model_name Fusion \
  --img_size 256 \
  --base_lr 0.01


Key parameters:

--n_gpu: number of GPUs to use

--root_path: path to the training dataset

--test_path: path to the testing dataset

--module: model architecture (e.g., HAformerSpatialFrequency)

--dataset: dataset type (here: Multiphase)

--eval_interval: evaluation frequency in epochs

--max_epochs: total training epochs

--batch_size: samples per batch

--model_name: checkpoint saving name

--img_size: input image dimensions

--base_lr: base learning rate

📊 Training Outputs

All outputs are saved to:

multi_phase/multi_phase/model_out/


Included:

Best model weights

Training and validation curves

Evaluation metrics: DSC, Jaccard, HD95, ASSD

After training, the saved models can be directly used for inference, evaluation, and visualization.











