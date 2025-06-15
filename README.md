# alpha-beta-crown

**α, β-CROWN 기반 FashionMNIST MLP Robustness Verification**

---

## 📚 프로젝트 개요

- **목표:**  
  α, β-CROWN 도구를 사용해 외부 신경망 모델(MLP, FashionMNIST)에 대한 adversarial robustness(공식 강건성)를 자동 검증합니다.

---

## 📁 Project Structure
```
.
alpha-beta-CROWN/
├── complete_verifier/
│ ├── abcrown.py
│ ├── custom/
│ │ └── fashion_model_data.py # MLP모델/데이터로더
│ ├── models/
│ │ └── fashionmnist_mlp.pth # 학습된 모델 파라미터
│ ├── exp_configs/
│ │ └── fashionmnist_mlp.yaml # 실험 YAML
│ ├── fashionmnist_train.py # train(.pth weight 저장)
│ └── README.md
```

---

##  ⚙️ Setup

### 1. Clone the repository

```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
```

### 2. Install dependencies

```bash
# Remove the old environment, if necessary.
conda deactivate; conda env remove --name alpha-beta-crown
# install all dependents into the alpha-beta-crown environment
conda env create -f complete_verifier/environment.yaml --name alpha-beta-crown
# activate the environment
conda activate alpha-beta-crown
```

### 3. Set to sturture
```
Enter the file
- fashion_model_data.py
- fashionmnist_mlp.yaml
- fashionmnist_train.py
in the format of the structure.
```

---

## 🚀 Usage

### 1. run fashionmnist_train.py
```bash
cd complete_verifier
python fashionmnist_train.py
```

### 2. run abcrown.py
```bash
cd complete_verifier
python abcrown.py --config exp_configs/fashionmnist_mlp.yaml 
```
```bash
sbatch jupyter.sh
- The results are stored in STDOUT and the log is stored in STDERR.
```

---

## 📊 Results 

```
############# Summary #############
Final verified acc: 60.0% (total 10 examples)
Problem instances count: 10 , total verified (safe/unsat): 6 , total falsified (unsafe/sat): 4 , timeout: 0
mean time for ALL instances (total 10):0.7734712495390044, max time: 4.262286901473999
mean time for verified SAFE instances(total 6): 1.1444157759348552, max time: 4.262286901473999
mean time for verified (SAFE + UNSAFE) instances (total 10): 0.7734720230102539, max time: [0.8399333953857422, 0.008143901824951172, 0.011816978454589844, 0.008331298828125]
mean time for verified UNSAFE instances (total 4): 0.21705639362335205, max time: 0.8399333953857422
unsafe-pgd (total 4), index: [0, 4, 7, 8]
safe-incomplete (total 4), index: [1, 2, 3, 5]
safe (total 2), index: [6, 9]
```




