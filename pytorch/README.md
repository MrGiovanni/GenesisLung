# Genesis Lung - Official PyTorch Implementation

### 0. Create a visual environment

```bash
mkdir environments
cd environments
python3 -m venv pytorch
source /data/jliang12/zzhou82/environments/pytorch/bin/activate
```

### 1. Clone the repository


```bash
git clone https://github.com/MrGiovanni/GenesisLung
cd GenesisLung/
pip install torch torchvision torchaudio # refer to https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

### 2. Pre-train Genesis Lung

```bash
cd pytorch/
mkdir logs pair_samples pretrained_weights
```

#### Run in ASU Agave Cluster

```bash
sbatch --error=logs/genesis_lung.out --output=logs/genesis_lung.out run.sh /data/jliang12/zzhou82/holy_grail None
```

#### Run in JLiang lab GPU machines

```bash
python -W ignore genesis_lung.py --gpu 0 --data None --weights None
```