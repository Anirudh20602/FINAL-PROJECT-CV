# Low-Light Pedestrian Risk Detection from Thermal + RGB Fusion

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A risk-aware perception system combining visible (RGB) and thermal (LWIR) imagery for robust pedestrian detection and collision risk assessment in low-light and adverse weather conditions.

## 🎯 Project Overview

This project implements a multimodal fusion architecture that:
- **Detects pedestrians** in challenging lighting conditions (night, fog, rain)
- **Assesses collision risk** using calibrated confidence scores
- **Predicts crossing intent** (future work)
- **Generalizes across datasets** (KAIST, LLVIP, FLIR ADAS)

### Key Features

✅ **Dual-Stream Architecture**: ResNet-34 backbone with separate RGB and Thermal encoders  
✅ **Bi-Directional Cross-Attention**: Novel fusion mechanism for modality integration  
✅ **Risk Calibration**: Temperature scaling to reduce Expected Calibration Error (ECE)  
✅ **Multi-Dataset Support**: KAIST, LLVIP, and FLIR ADAS datasets  
✅ **SOTA Performance**: 0.64 mAP@0.5 on KAIST night scenes with ECE of 0.041  

## 📊 Performance

| Model | Dataset | Scene | mAP@0.5 | ECE |
|-------|---------|-------|---------|-----|
| RGB-only | KAIST | Night | 0.41 | — |
| Thermal-only | KAIST | Night | 0.55 | — |
| Early Fusion (4-ch) | KAIST | Night | 0.59 | 0.093 |
| **Mid Fusion (Bi-XAttn)** | KAIST | Night | **0.64** | **0.041** |

## 🏗️ Architecture

```
Input: RGB (3-ch) + Thermal (1-ch)
         ↓
    ┌────────────────┐
    │ Dual-Stream    │
    │ ResNet-34      │
    │ Backbone       │
    └────────────────┘
         ↓
    ┌────────────────┐
    │ Bi-Directional │
    │ Cross-Attention│
    │ Fusion Neck    │
    └────────────────┘
         ↓
    ┌────────────────┐
    │ Detection Head │
    │ + Risk Head    │
    └────────────────┘
         ↓
    Output: Bboxes + Risk Scores
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/low-light-pedestrian-risk-detection.git
cd low-light-pedestrian-risk-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Dataset Preparation

1. **Download datasets** (KAIST, LLVIP, FLIR ADAS)
2. **Organize** into `data/` directory structure
3. **Preprocess** and align RGB-Thermal pairs

```bash
# Download KAIST dataset
python scripts/download_datasets.py --dataset kaist --output data/kaist/raw

# Preprocess and align
python scripts/preprocess_data.py --dataset kaist --align --output data/kaist/processed
```

### Training

```bash
# Train baseline RGB-only model
python scripts/train.py --config configs/baseline_rgb.yaml

# Train early fusion model
python scripts/train.py --config configs/early_fusion.yaml

# Train mid-fusion model with Bi-XAttn
python scripts/train.py --config configs/mid_fusion.yaml
```

### Evaluation

```bash
# Evaluate on KAIST test set
python scripts/evaluate.py --config configs/mid_fusion.yaml --checkpoint experiments/mid_fusion/best.pth

# Cross-dataset evaluation
python scripts/evaluate.py --config configs/mid_fusion.yaml --checkpoint experiments/mid_fusion/best.pth --dataset llvip
```

### Inference

```bash
# Run inference on images
python scripts/inference.py --config configs/mid_fusion.yaml --checkpoint experiments/mid_fusion/best.pth --input path/to/images --output results/
```

## 📁 Project Structure

```
low-light-pedestrian-risk-detection/
├── data/                      # Datasets
│   ├── kaist/
│   ├── llvip/
│   └── flir/
├── src/                       # Source code
│   ├── data/                  # Data processing
│   ├── models/                # Model architectures
│   ├── training/              # Training utilities
│   ├── evaluation/            # Evaluation metrics
│   └── utils/                 # Helper functions
├── configs/                   # Configuration files
├── scripts/                   # Training/evaluation scripts
├── notebooks/                 # Jupyter notebooks
├── experiments/               # Experiment outputs
├── docs/                      # Documentation
└── tests/                     # Unit tests
```

## 🔬 Technical Details

### RGB-Thermal Alignment
- **Method**: SIFT feature detection + RANSAC homography estimation
- **Accuracy**: <1.8px mean alignment error
- **Output**: Aligned 4-channel tensors [RGB, Thermal]

### Bi-Directional Cross-Attention
- **Query**: RGB features → Attend to Thermal features
- **Key/Value**: Thermal features → Attend to RGB features
- **Mechanism**: Multi-head attention with residual connections
- **Benefit**: Adaptive modality fusion based on scene context

### Risk Calibration
- **Method**: Temperature scaling on validation set
- **Input**: Bounding box geometry + centroid velocity + fused embeddings
- **Output**: Calibrated risk score [0, 1]
- **Metric**: Expected Calibration Error (ECE)

## 📚 Datasets

### KAIST Multispectral Pedestrian Dataset
- 95,000 RGB-Thermal image pairs
- 640×480 resolution @ 20 fps
- Day/night scenes from moving vehicle
- 100,000+ bounding box annotations

### LLVIP (Low-Light Visible-Infrared Paired Dataset)
- 30,976 RGB-Thermal image pairs
- Focus on low-light conditions
- Diverse urban scenarios

### FLIR ADAS Dataset
- 10,228 thermal images
- Real-world driving scenarios
- Pre-annotated pedestrians

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ scripts/ tests/
flake8 src/ scripts/ tests/
```

### Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

## 📈 Roadmap

- [x] RGB-Thermal alignment pipeline
- [x] Baseline models (RGB-only, Thermal-only)
- [x] Early fusion architecture
- [x] Mid-fusion with Bi-XAttn
- [x] Risk calibration
- [ ] Intent classification head
- [ ] Domain adaptation (LLVIP→FLIR)
- [ ] Ablation studies
- [ ] Video demo generation
- [ ] Real-time inference optimization

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@misc{krishna2025lowlight,
  title={Low-Light Pedestrian Risk Detection from Thermal + RGB Fusion},
  author={Krishna, Anirudh},
  year={2025},
  institution={University of Maryland, College Park}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **KAIST Multispectral Pedestrian Dataset** by Soonmin Hwang et al.
- **LLVIP Dataset** by Xinyu Jia et al.
- **FLIR ADAS Dataset** by FLIR Systems
- **MMDetection** framework by OpenMMLab
- **PyTorch** deep learning framework

## 👤 Author

**Anirudh Krishna**  
Master of Science in Applied Machine Learning  
University of Maryland, College Park  
Course: MSML 640 - Computer Vision  
Instructor: Sujeong Kim

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact via email.

---

**Status**: 🟢 Active Development | **Last Updated**: December 2025
