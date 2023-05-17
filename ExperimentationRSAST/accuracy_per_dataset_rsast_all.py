# %% [markdown]
# ## Experimentation Accuracy RSAST per Dataset:
# 

# %%
#configure directory to import sast libraries
import sys 
import os 
#add sast library path
file_path = os.path.dirname(os.getcwd())+"/sast"

#file_path = r"C:\Users\Public\random_sast\sast"
sys.path.append(file_path)

file_path = os.path.dirname(os.getcwd())+"\sast"


#file_path = r"C:\Users\Public\random_sast\sast"
sys.path.append(file_path)

file_path = os.getcwd()+"/sast"


#file_path = r"C:\Users\Public\random_sast\sast"
sys.path.append(file_path)

file_path = os.getcwd()+"\sast"


#file_path = r"C:\Users\Public\random_sast\sast"
sys.path.append(file_path)

#add cd_diagram library path
file_path = os.path.dirname(os.getcwd())+"\cd_diagram"

#file_path = r"C:\Users\Public\random_sast\cd_diagram"
sys.path.append(file_path)


file_path = os.path.dirname(os.getcwd())+"/cd_diagram"
#file_path = r"C:\Users\Public\random_sast\sast"
sys.path.append(file_path)

file_path = os.getcwd()+"\cd_diagram"
#file_path = r"C:\Users\Public\random_sast\sast"
sys.path.append(file_path)

file_path = os.getcwd()+"/cd_diagram"
#file_path = r"C:\Users\Public\random_sast\sast"
sys.path.append(file_path)

sys.path
#os.chdir(os.getcwd()+"/ExperimentationRSAST")
os.getcwd()


# %%
from sast import *
from sktime.datasets import load_UCR_UEA_dataset, tsc_dataset_names
from sktime.classification.kernel_based import RocketClassifier
import time
import pandas as pd

# %% [markdown]
# ### Select Datasets for hypertunning RSAST

# %% [markdown]
# It is runned RSAST in a set of UCR datasets with a predefined number of runs ("runs"). Then, it is selected a range ("range_total") between [1, 10, 30 ,50,100] for the selected dataset.

# %%

ds_sens=pd.read_excel("DataSetsUCLASummary.xlsx")

ds_sens=ds_sens[ds_sens['N RUNS S17_SAST_DS'].isna()]
ds_sens=ds_sens[ds_sens['USED SAST']=="Y"]
ds_sens=ds_sens.Name.unique()
len(ds_sens)
'''
ds_sens = tsc_dataset_names.univariate_equal_length
#list_remove=["SmoothSubspace","Chinatown","ItalyPowerDemand","SyntheticControl","SonyAIBORobotSurface2","DistalPhalanxOutlineAgeGroup","DistalPhalanxOutlineCorrect","GunPoint","Fungi","Coffee","ShapeletSim"]
list_remove=ds_sens.Name.unique()[1:29]
# using set() to perform task
set1 = set(ds_sens)
set2 = set(list_remove)

ds_sens = list(set1 - set2)
'''






#ds_sens1 = ['SmoothSubspace', 'Car', 'ECG5000']

#ds_sens2 = ['ToeSegmentation2', 'ItalyPowerDemand','Crop']

#ds_sens=ds_sens1

#ds_sens = [ 'SmoothSubspace']


max_ds=len(ds_sens) #exploring dataset in UEA & UCR Time Series Classification Repository
print(max_ds)
print(ds_sens)

# %%
#define numbers of runs of the experiment
runs = 5

#define range for number of random points 
range_rpoint = [10, 30]

#define range for number of intances per class
range_nb_inst_per_class=[1, 10]


#define lenght method
len_methods = ["both", "Max PACF", "None"]


not_found_ds =[]

for ds in ds_sens:

    try:    
        X_train, y_train = load_UCR_UEA_dataset(name=ds, extract_path='data', split="train", return_type="numpy2d")
        X_test, y_test = load_UCR_UEA_dataset(name=ds, extract_path='data', split="test", return_type="numpy2d")
        print("ds="+ds)
    except:
        print("not found ds="+ds)
        not_found_ds.append(ds)
        continue

    for i in range(runs):
        df_result = {}
        list_score = []
        list_overall_time = []
        list_cweight_time = []
        list_fsubsequence_time = []
        list_tdataset_time = []
        list_tclassifier_time = []
        list_dataset = []
        list_hyperparameter = []
        list_method = []
        list_rpoint = []
        list_nb_per_class = []
        list_len_method = []
        for len_m in len_methods:
            for p in range_rpoint:
                for g in range_nb_inst_per_class:
                    if p ==0:
                        p=1
                    if g ==0:
                        g=1
                    if len_m=="both":
                        len_m_corrected="ACF&PACF" 
                    else:
                        len_m_corrected=len_m
                    print(len_m_corrected+": n_random_points="+str(p)+" nb_inst_per_class="+str(g))
                    start = time.time()
                    random_state = None
                    rsast_ridge = RSAST(n_random_points=p,nb_inst_per_class=g, len_method=len_m)
                    rsast_ridge.fit(X_train, y_train)
                    end = time.time()
                    
                    
                    list_score.append(rsast_ridge.score(X_test,y_test))

                    list_overall_time.append(end-start)
                    list_cweight_time.append(rsast_ridge.time_calculating_weights)
                    list_fsubsequence_time.append(rsast_ridge.time_creating_subsequences)
                    list_tdataset_time.append(rsast_ridge.transform_dataset)
                    list_tclassifier_time.append(rsast_ridge.time_classifier)

                    list_dataset.append(ds)
                    list_hyperparameter.append(len_m_corrected+": n_random_points="+str(p)+" nb_inst_per_class="+str(g))
                    list_rpoint.append(str(p))
                    list_nb_per_class.append(str(g))
                    list_method.append("Rsast")
                    list_len_method.append(len_m_corrected)
                    
        df_result['accuracy']=list_score
        df_result['time']=list_overall_time
        df_result['cweights_time']=list_cweight_time
        df_result['fsubsequence_time']=list_fsubsequence_time
        df_result['tdataset_time']=list_tdataset_time
        df_result['tclassifier_time']=list_tclassifier_time
        df_result['dataset_name']=list_dataset
        df_result['classifier_name']=list_hyperparameter
        df_result['rpoint']=list_rpoint
        df_result['nb_per_class']=list_nb_per_class
        df_result['method']=list_method
        df_result['len_method']=list_len_method
        df_result=pd.DataFrame(df_result)
        # export a overall dataset with results
        df_result.to_csv("results_accuracy_per_ds/df_all_overall_tunning_"+str(ds)+str(i+1)+"_norepTSRP.csv") 


