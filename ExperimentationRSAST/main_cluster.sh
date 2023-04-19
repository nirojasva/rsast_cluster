#!/bin/bash
 
#===============================================================================
# exemples d'options
 
#SBATCH --partition=normal    # choix de la partition où soumettre le job
#SBATCH --time=10:0           # temps max alloué au job (format = m:s ou h:m:s ou j-h:m:s)
#SBATCH --ntasks=1            # nb de tasks total pour le job
#SBATCH --cpus-per-task=1     # 1 seul CPU pour une task
#SBATCH --mem=1000            # mémoire nécessaire (par noeud) en Mo
 
#===============================================================================
#exécution du programme (remplacer exe par le nom du programme
# ou la ligne de commande à exécuter)
python "accuracy_per_dataset_rsast (acf_pacf_10000).py"