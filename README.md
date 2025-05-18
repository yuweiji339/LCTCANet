
# LCTCANet

## Overview

**LCTCANet** (Lightweight CNN-Transformer Context-Aware Network) is a super-resolution model designed for remote sensing imagery. It integrates both local feature extraction and global context modeling to reconstruct high-fidelity, high-resolution images while maintaining efficiency suitable for edge and mobile deployment.

### Key Components

- **GlobalContextualLocalBlock (GCLB)**: Hybrid module combining multi-head self-attention with convolutional encoding for long-range dependency modeling and fine-grained texture recovery.  
- **EdgeStructureFusionBlock (ESFB)**: Dual-branch block with edge-aware attention and structural enhancement to fuse detailed edges and contextual structure.  
- **DeepFeatureExtractionBlock (DFEB)**: Stacked ESFB with residual connections to progressively refine features.  

## Features

- **Lightweight**: Reduced parameters and FLOPs for real-world deployment.  
- **Context-Aware**: Joint local-global feature fusion.  
- **Modular**: Easily customizable block counts and dimensions.  
- **Easy Integration**: Compatible with [BasicSR](https://github.com/XPixelGroup/BasicSR) registry system.  

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LCTCANet.git
cd LCTCANet

# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Install as a package
pip install -e .

#Contributing
Please read CONTRIBUTING.md for details on the code of conduct and the process for submitting pull requests.

#License
This project is licensed under the MIT License — see the LICENSE file for details.
