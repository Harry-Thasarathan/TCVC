## TCVC: Temporally Coherent Video Colorization
[ArXiv](https://arxiv.org/abs/1904.09527) | [BibTex](#citation)

### Introduction:
We develop a new approach to automatically colorize line art/greyscale anime episode frames to a temporally coherent video using an end-to-end GAN based learning framework. In the animation industry, heavy workloads and overtime are the norm for young artists. When between 3000 to 10000 frames need to be individually animated for a single episode, the work needs to be distributed and outsourced which leads to slow turn around. This method is our initial foray at producing an algorithm that could benefit animation studios and reduce workloads. We mimic lineart using Canny edge detection and adapt a training scheme to preserve temporal information between generated frames to mitigate the flicker effect that exists when using GANs to generate individual frames for video. Detailed information of our work can be found in our [paper](https://arxiv.org/abs/1904.09527).

## Prerequisites
- Python 3
- PyTorch 1.0
- NVIDIA GPU + CUDA cuDNN

## Installation
- Clone this repo:
```bash
git clone https://github.com/harry-thasarathan/TCVC.git
cd TCVC
```
- Install PyTorch and dependencies from http://pytorch.org
- Install python requirements:
```bash
pip install -r requirements.txt
```
