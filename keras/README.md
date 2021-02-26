# Genesis Lung - Official Keras Implementation

### 1. Clone the repository


```bash
git clone https://github.com/MrGiovanni/GenesisLung
cd GenesisLung/
pip install -r requirements.txt
```

### 2. Pre-train Genesis Lung

```bash
cd keras/
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