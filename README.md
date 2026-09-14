# HyCoSeq: Contextual Hyperbolic Representation Learning for Genomic Sequences

This repository provides reproducible code for the main analyses described in Chenhao Zeng, et al. *HyCoSeq: Contextual Hyperbolic Representation Learning for Genomic Sequences*. 

## Project Structure

```text
hycseq/
├── data.py
├── engine.py
├── network.py
├── recurrent.py
├── residual.py
├── runtime.py
├── settings.py
└── tokens.py
geometry/
├── geoopt/
└── hyperboloid/
recipes/
run_experiment.py
```

## Installation

### Requirements

- Python\>= 3.8 

```bash
conda create -n hycseq python=3.8
conda activate hycseq
```

- PyTorch 1.13 with CUDA 11.7

```bash
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

- Install the remaining dependencies

```bash
pip install -r requirements.txt
```

## Citation

If you use the code (partial or all) in your research, please cite the following manuscript:
Chenhao Zeng, et al. *HyCoSeq: Contextual Hyperbolic Representation Learning for Genomic Sequences*. 
