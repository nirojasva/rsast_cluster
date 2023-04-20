#!/bin/bash
#SBATCH --job-name=rsast_project

#SBATCH --partition=normal    # choix de la partition où soumettre le job
#SBATCH --time=1:00:00          # temps max alloué au job (format = m:s ou h:m:s ou j-h:m:s)
#SBATCH --ntasks=5            # nb de tasks total pour le job
#SBATCH --cpus-per-task=113     # 1 seul CPU pour une task
#SBATCH --mem=400            # mémoire nécessaire (par noeud) en Mo
 

# Activate the Python virtual environment
source rsast_env/bin/activate

# Execute the Python script
cd ~/rsast_cluster/ExperimentationRSAST
python accuracy_per_dataset_rsast_all.py 

wait

echo "All jobs complete"