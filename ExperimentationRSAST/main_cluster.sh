#!/bin/bash
#SBATCH --job-name=rsast_project
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=400
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=400G

# Activate the Python virtual environment
source rsast_env/bin/activate

# Execute the Python script
cd ~/rsast_cluster/ExperimentationRSAST
python accuracy_per_dataset_rsast.py &
python 'accuracy_per_dataset_rsast (extra).py' &
python 'accuracy_per_dataset_rsast (extra_acf_pacf_10000).py' &
python 'accuracy_per_dataset_rsast (acf_pacf_10000).py' &

wait

echo "All jobs complete"