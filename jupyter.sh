#!/bin/bash
#SBATCH --nodes=1 # 자동으로 알아서 컴퓨터가 배정해줌, 노드 한개만을 쓰겠다는 뜻 / Heavy한 JOB이 아니면 노드는 한개로도 충분 / 총 필요한 노드 수를 지정하는 명령어 
#SBATCH --partition=gpu1 # 어떤 파티션을 쓸 것인지 선택 ( 위 파티션을 참조하여 선택 )
#SBATCH --cpus-per-task=1 # 노드당 CPU나 GPU 코어를 몇 개를 쓸 것인지 지정 ( # of Cores/node 참조 )
#SBATCH --gres=gpu:4 # GPU를 몇 개를 쓸 것인지 지정하기 위한 옵션
#SBATCH --job-name=ABC # 작업 이름 지정 , 필수적이지는 않음   
#SBATCH -o ./STDOUT/jupyter.%N.%j.out  # STDOUT ( Standard Output ) 
#SBATCH -e ./STDERR/jupyter.%N.%j.err  # STDERR ( Standard Error ) 

echo "start at:" `date` # 접속한 날짜 표기
echo "node: $HOSTNAME" # 접속한 노드 번호 표기 
echo "jobid: $SLURM_JOB_ID" # jobid 표기 

# GPU 환경을 이용하고 싶은 경우에만 해당! 그렇지 않은 경우 해당 명령어들은 지우셔도 무관합니다.
# module avail CUDA # CUDA 어떤 버전들이 설치되어있는지 확인하는 방법 

module load cuda/12.2.1

# nvidia-smi

# nvcc --version  # CUDA 버전 확인

# python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# python -c "import torch; print(torch.version.cuda)"

# module list

# pip list

######################################################################################

cd complete_verifier

python abcrown.py --config exp_configs/fashionmnist_mlp.yaml 
