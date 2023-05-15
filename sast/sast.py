# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC

from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.pipeline import Pipeline

#from sktime.utils.data_processing import from_2d_array_to_nested
from sktime.transformations.panel.rocket import Rocket

from numba import njit, prange

from mass_ts import *

import pandas as pd

from scipy.stats import f_oneway
from statsmodels.tsa.stattools import acf,pacf
import warnings
import time

import os
from operator import itemgetter

def from_2d_array_to_nested(
    X, index=None, columns=None, time_index=None, cells_as_numpy=False
):
    """Convert 2D dataframe to nested dataframe.
    Convert tabular pandas DataFrame with only primitives in cells into
    nested pandas DataFrame with a single column.
    Parameters
    ----------
    X : pd.DataFrame
    cells_as_numpy : bool, default = False
        If True, then nested cells contain NumPy array
        If False, then nested cells contain pandas Series
    index : array-like, shape=[n_samples], optional (default = None)
        Sample (row) index of transformed DataFrame
    time_index : array-like, shape=[n_obs], optional (default = None)
        Time series index of transformed DataFrame
    Returns
    -------
    Xt : pd.DataFrame
        Transformed DataFrame in nested format
    """
    if (time_index is not None) and cells_as_numpy:
        raise ValueError(
            "`Time_index` cannot be specified when `return_arrays` is True, "
            "time index can only be set to "
            "pandas Series"
        )
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    container = np.array if cells_as_numpy else pd.Series

    # for 2d numpy array, rows represent instances, columns represent time points
    n_instances, n_timepoints = X.shape

    if time_index is None:
        time_index = np.arange(n_timepoints)
    kwargs = {"index": time_index}

    Xt = pd.DataFrame(
        pd.Series([container(X[i, :], **kwargs) for i in range(n_instances)])
    )
    if index is not None:
        Xt.index = index
    if columns is not None:
        Xt.columns = columns
    return Xt


@njit(fastmath=True)
def znormalize_array(arr):
    m = np.mean(arr)
    s = np.std(arr)

    # s[s == 0] = 1 # avoid division by zero if any

    return (arr - m) / (s + 1e-8)
    # return arr


@njit(fastmath=False)
def apply_kernel(ts, arr):
    d_best = np.inf  # sdist
    m = ts.shape[0]
    kernel = arr[~np.isnan(arr)]  # ignore nan

    # profile = mass2(ts, kernel)
    # d_best = np.min(profile)

    l = kernel.shape[0]
    for i in range(m - l + 1):
        d = np.sum((znormalize_array(ts[i:i+l]) - kernel)**2)
        if d < d_best:
            d_best = d

    return d_best


@njit(parallel=True, fastmath=True)
def apply_kernels(X, kernels):
    nbk = len(kernels)
    out = np.zeros((X.shape[0], nbk), dtype=np.float32)
    for i in prange(nbk):
        k = kernels[i]
        for t in range(X.shape[0]):
            ts = X[t]
            out[t][i] = apply_kernel(ts, k)
    return out


class SAST(BaseEstimator, ClassifierMixin):

    def __init__(self, cand_length_list, shp_step=1, nb_inst_per_class=1, random_state=None, classifier=None):
        super(SAST, self).__init__()
        self.cand_length_list = cand_length_list
        self.shp_step = shp_step
        self.nb_inst_per_class = nb_inst_per_class
        self.kernels_ = None
        self.kernel_orig_ = None  # not z-normalized kernels
        self.kernels_generators_ = {}
        self.random_state = np.random.RandomState(random_state) if not isinstance(
            random_state, np.random.RandomState) else random_state

        self.classifier = classifier

    def get_params(self, deep=True):
        return {
            'cand_length_list': self.cand_length_list,
            'shp_step': self.shp_step,
            'nb_inst_per_class': self.nb_inst_per_class,
            'classifier': self.classifier
        }

    def init_sast(self, X, y):

        self.cand_length_list = np.array(sorted(self.cand_length_list))

        assert self.cand_length_list.ndim == 1, 'Invalid shapelet length list: required list or tuple, or a 1d numpy array'

        if self.classifier is None:
            self.classifier = RandomForestClassifier(
                min_impurity_decrease=0.05, max_features=None)

        classes = np.unique(y)
        self.num_classes = classes.shape[0]

        candidates_ts = []
        for c in classes:
            X_c = X[y == c]

            # convert to int because if self.nb_inst_per_class is float, the result of np.min() will be float
            cnt = np.min([self.nb_inst_per_class, X_c.shape[0]]).astype(int)
            choosen = self.random_state.permutation(X_c.shape[0])[:cnt]
            candidates_ts.append(X_c[choosen])
            self.kernels_generators_[c] = X_c[choosen]

        candidates_ts = np.concatenate(candidates_ts, axis=0)

        self.cand_length_list = self.cand_length_list[self.cand_length_list <= X.shape[1]]

        max_shp_length = max(self.cand_length_list)

        n, m = candidates_ts.shape

        n_kernels = n * np.sum([m - l + 1 for l in self.cand_length_list])

        self.kernels_ = np.full(
            (n_kernels, max_shp_length), dtype=np.float32, fill_value=np.nan)
        self.kernel_orig_ = []

        k = 0

        for shp_length in self.cand_length_list:
            for i in range(candidates_ts.shape[0]):
                for j in range(0, candidates_ts.shape[1] - shp_length + 1, self.shp_step):
                    end = j + shp_length
                    can = np.squeeze(candidates_ts[i][j: end])
                    self.kernel_orig_.append(can)
                    self.kernels_[k, :shp_length] = znormalize_array(can)

                    k += 1
        


    def fit(self, X, y):

        X, y = check_X_y(X, y)  # check the shape of the data

        # randomly choose reference time series and generate kernels
        self.init_sast(X, y)

        # subsequence transform of X
        X_transformed = apply_kernels(X, self.kernels_)

        self.classifier.fit(X_transformed, y)  # fit the classifier

        return self

    def predict(self, X):

        check_is_fitted(self)  # make sure the classifier is fitted

        X = check_array(X)  # validate the shape of X

        # subsequence transform of X
        X_transformed = apply_kernels(X, self.kernels_)

        return self.classifier.predict(X_transformed)

    def predict_proba(self, X):
        check_is_fitted(self)  # make sure the classifier is fitted

        X = check_array(X)  # validate the shape of X

        # subsequence transform of X
        X_transformed = apply_kernels(X, self.kernels_)

        if isinstance(self.classifier, LinearClassifierMixin):
            return self.classifier._predict_proba_lr(X_transformed)
        return self.classifier.predict_proba(X_transformed)


class SASTEnsemble(BaseEstimator, ClassifierMixin):

    def __init__(self, cand_length_list, shp_step=1, nb_inst_per_class=1, random_state=None, classifier=None, weights=None, n_jobs=None):
        super(SASTEnsemble, self).__init__()
        self.cand_length_list = cand_length_list
        self.shp_step = shp_step
        self.nb_inst_per_class = nb_inst_per_class
        self.classifier = classifier
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.saste = None

        self.weights = weights

        assert isinstance(self.classifier, BaseEstimator)

        self.init_ensemble()

    def init_ensemble(self):
        estimators = []
        for i, candidate_lengths in enumerate(self.cand_length_list):
            clf = clone(self.classifier)
            sast = SAST(cand_length_list=candidate_lengths,
                        nb_inst_per_class=self.nb_inst_per_class,
                        random_state=self.random_state,
                        shp_step=self.shp_step,
                        classifier=clf)
            estimators.append((f'sast{i}', sast))

        self.saste = VotingClassifier(
            estimators=estimators, voting='soft', n_jobs=self.n_jobs, weights=self.weights)

    def fit(self, X, y):
        self.saste.fit(X, y)
        return self

    def predict(self, X):
        return self.saste.predict(X)

    def predict_proba(self, X):
        return self.saste.predict_proba(X)


class RocketClassifier:
    def __init__(self, num_kernels=10000, normalise=True, random_state=None, clf=None, lr_clf=True):
        rocket = Rocket(num_kernels=num_kernels,
                        normalise=normalise, random_state=random_state)
        clf = RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 10)) if clf is None else clf
        self.model = Pipeline(steps=[('rocket', rocket), ('clf', clf)])
        # False if the classifier has the method predict_proba, otherwise False
        self.lr_clf = lr_clf

    def fit(self, X, y):
        self.model.fit(from_2d_array_to_nested(X), y)

    def predict(self, X):
        return self.model.predict(from_2d_array_to_nested(X))

    def predict_proba(self, X):
        X_df = from_2d_array_to_nested(X)
        if not self.lr_clf:
            return self.model.predict_proba(X_df)
        X_transformed = self.model['rocket'].transform(X_df)
        return self.model['clf']._predict_proba_lr(X_transformed)


class RSAST(BaseEstimator, ClassifierMixin):

    def __init__(self,n_random_points=10, nb_inst_per_class=1, len_method="both", random_state=None, classifier=None, sel_inst_wrepl=False,sel_randp_wrepl=False, half_instance=False, half_len=False ):
        super(RSAST, self).__init__()
        self.n_random_points = n_random_points
        self.nb_inst_per_class = nb_inst_per_class
        self.len_method = len_method
        self.random_state = np.random.RandomState(random_state) if not isinstance(
            random_state, np.random.RandomState) else random_state
        self.classifier = classifier
        self.cand_length_list = None
        self.kernels_ = None
        self.kernel_orig_ = None  # not z-normalized kernels
        self.kernels_generators_ = {}
        self.sel_inst_wrepl=sel_inst_wrepl
        self.sel_randp_wrepl=sel_randp_wrepl
        self.half_instance=half_instance
        self.half_len=half_len
        self.time_calculating_weights = None
        self.time_creating_subsequences = None
        self.time_transform_dataset = None
        self.time_classifier = None

    def get_params(self, deep=True):
        return {
            'len_method': self.len_method,
            'n_random_points': self.n_random_points,
            'nb_inst_per_class': self.nb_inst_per_class,
            'sel_inst_wrepl':self.sel_inst_wrepl,
            'sel_randp_wrepl':self.sel_randp_wrepl,
            'half_instance':self.half_instance,
            'half_len':self.half_len,        
            'classifier': self.classifier,
            'cand_length_list': self.cand_length_list
        }

    def init_sast(self, X, y):
        #0- initialize variables and convert values in "y" to string
        start = time.time()
        y=np.asarray([str(x_s) for x_s in y])
        
        self.cand_length_list = {}
        self.kernel_orig_ = []
        
        list_kernels =[]
        candidates_ts = []
        
        statistic_per_class= {}
        n = []
        classes = np.unique(y)
        self.num_classes = classes.shape[0]
        m_kernel = 0

        #1--calculate ANOVA per each time t throught the lenght of the TS
        for i in range (X.shape[1]):
            for c in classes:
                assert len(X[np.where(y==c)[0]][:,i])> 0, 'Time t without values in TS'
                statistic_per_class[c]=X[np.where(y==c)[0]][:,i]
            
            statistic_per_class=pd.Series(statistic_per_class)
            # Calculate t-statistic and p-value
            t_statistic, p_value = f_oneway(*statistic_per_class)
            
            # Interpretation of the results
            # if p_value < 0.05: " The means of the populations are significantly different."
            if np.isnan(p_value):
                n.append(0)
            else:
                n.append(1-p_value)
        end = time.time()
        self.time_calculating_weights = end-start


        #2--calculate PACF and ACF for each TS chossen in each class
        start = time.time()
        for i, c in enumerate(classes):
            X_c = X[y == c]
            if self.half_instance==True:
                cnt = np.max([X_c.shape[0]//2, 1]).astype(int)
                self.nb_inst_per_class=cnt
            else:
                cnt = np.min([self.nb_inst_per_class, X_c.shape[0]]).astype(int)
            #set if the selection of instances is with replacement (if false it is not posible to select the same intance more than one)
            if self.sel_inst_wrepl ==False:
                choosen = self.random_state.permutation(X_c.shape[0])[:cnt]
            else:
                choosen = self.random_state.choice(X_c.shape[0], cnt)
            candidates_ts.append(X_c[choosen])
            self.kernels_generators_[c] = X_c[choosen]
            
            
            for rep, idx in enumerate(choosen):
                self.cand_length_list[c+","+str(idx)+","+str(rep)] = []
                non_zero_acf=[]
                if (self.len_method == "both" or self.len_method == "ACF" or self.len_method == "Max ACF") :
                #2.1-- Compute Autorrelation per object
                    acf_val, acf_confint = acf(X_c[idx], nlags=len(X_c[idx])-1,  alpha=.05)
                    prev_acf=0    
                    for j, conf in enumerate(acf_confint):

                        if(3<=j and (0 < acf_confint[j][0] <= acf_confint[j][1] or acf_confint[j][0] <= acf_confint[j][1] < 0) ):
                            #Consider just the maximum ACF value
                            if prev_acf!=0 and self.len_method == "Max ACF":
                                non_zero_acf.remove(prev_acf)
                                self.cand_length_list[c+","+str(idx)+","+str(rep)].remove(prev_acf)
                            non_zero_acf.append(j)
                            self.cand_length_list[c+","+str(idx)+","+str(rep)].append(j)
                            prev_acf=j        
                
                non_zero_pacf=[]
                if (self.len_method == "both" or self.len_method == "PACF" or self.len_method == "Max PACF"):
                    #2.2 Compute Partial Autorrelation per object
                    pacf_val, pacf_confint = pacf(X_c[idx], method="ols", nlags=(len(X_c[idx])//2) - 1,  alpha=.05)                
                    prev_pacf=0
                    for j, conf in enumerate(pacf_confint):

                        if(3<=j and (0 < pacf_confint[j][0] <= pacf_confint[j][1] or pacf_confint[j][0] <= pacf_confint[j][1] < 0) ):
                            #Consider just the maximum PACF value
                            if prev_pacf!=0 and self.len_method == "Max PACF":
                                non_zero_pacf.remove(prev_pacf)
                                self.cand_length_list[c+","+str(idx)+","+str(rep)].remove(prev_pacf)
                            
                            non_zero_pacf.append(j)
                            self.cand_length_list[c+","+str(idx)+","+str(rep)].append(j)
                            prev_pacf=j 
                            
                if (self.len_method == "all"):
                    self.cand_length_list[c+","+str(idx)+","+str(rep)].extend(np.arange(3,1+ len(X_c[idx])))
                
                #2.3-- Save the maximum autocorralated lag value as shapelet lenght 
                if len(non_zero_pacf)==0 and len(non_zero_acf)==0:
                    #chose a random lenght using the lenght of the time series (added 1 since the range start in 0)
                    rand_value= self.random_state.choice(len(X_c[idx]), 1)[0]+1
                    self.cand_length_list[c+","+str(idx)+","+str(rep)].extend([max(3,rand_value)])
                #elif len(non_zero_acf)==0:
                    #print("There is no AC in TS", idx, " of class ",c)
                #elif len(non_zero_pacf)==0:
                    #print("There is no PAC in TS", idx, " of class ",c)                 
                #else:
                    #print("There is AC and PAC in TS", idx, " of class ",c)

                #print("Kernel lenght list:",self.cand_length_list[c+","+str(idx)],"")
                 
                #remove duplicates for the list of lenghts
                self.cand_length_list[c+","+str(idx)+","+str(rep)]=list(set(self.cand_length_list[c+","+str(idx)+","+str(rep)]))
                for max_shp_length in self.cand_length_list[c+","+str(idx)+","+str(rep)]:
                    #2.4-- Choose randomly n_random_points point for a TS                
                    #2.5-- calculate the weights of probabilities for a random point in a TS
                    if sum(n) == 0 :
                        # Determine equal weights of a random point point in TS is there are no significant points
                        # print('All p values in One way ANOVA are equal to 0') 
                        weights = [1/len(n) for i in range(len(n))]
                        weights = weights[:len(X_c[idx])-max_shp_length +1]/np.sum(weights[:len(X_c[idx])-max_shp_length+1])
                    else: 
                        # Determine the weights of a random point point in TS (excluding points after n-l+1)
                        weights = n / np.sum(n)
                        weights = weights[:len(X_c[idx])-max_shp_length +1]/np.sum(weights[:len(X_c[idx])-max_shp_length+1])
                        
                    if self.half_len==True:
                        self.n_random_points=np.max([len(X_c[idx])//2, 1]).astype(int)
                    
                    if self.n_random_points > len(X_c[idx])-max_shp_length+1 and self.sel_randp_wrepl==False:
                        #set a upper limit for the posible of number of random points when selecting without replacement
                        limit_rpoint=len(X_c[idx])-max_shp_length+1
                        rand_point_ts = self.random_state.choice(len(X_c[idx])-max_shp_length+1, limit_rpoint, p=weights, replace=self.sel_randp_wrepl)
                    else:
                        rand_point_ts = self.random_state.choice(len(X_c[idx])-max_shp_length+1, self.n_random_points, p=weights, replace=self.sel_randp_wrepl)

                    for i in rand_point_ts:        
                        #2.6-- Extract the subsequence with that point
                        kernel = X_c[idx][i:i+max_shp_length].reshape(1,-1)
                        if m_kernel<max_shp_length:
                            m_kernel = max_shp_length            
                        list_kernels.append(kernel)
                        self.kernel_orig_.append(np.squeeze(kernel))
        
        #3--save the calculated subsequences
        candidates_ts = np.concatenate(candidates_ts, axis=0)
        n, m = candidates_ts.shape
        n_kernels = len (self.kernel_orig_)
        
        
        self.kernels_ = np.full(
            (n_kernels, m_kernel), dtype=np.float32, fill_value=np.nan)
        
        for k, kernel in enumerate(self.kernel_orig_):
            self.kernels_[k, :len(kernel)] = znormalize_array(kernel)
        
        end = time.time()
        self.time_creating_subsequences = end-start

    def fit(self, X, y):

        X, y = check_X_y(X, y)  # check the shape of the data

        # randomly choose reference time series and generate kernels
        self.init_sast(X, y)

        start = time.time()
        # subsequence transform of X
        X_transformed = apply_kernels(X, self.kernels_)
        end = time.time()
        self.transform_dataset = end-start
        
        if self.classifier is None:
            
            if X_transformed.shape[0]<=X_transformed.shape[1]:
                self.classifier=RidgeClassifierCV()
                print("RidgeClassifierCV:"+str("size training")+str(X_transformed.shape[0])+"<="+" kernels"+str(X_transformed.shape[1]))
            else: 
                print("LogisticRegressionCV:"+str("size training")+str(X_transformed.shape[0])+">"+" kernels"+str(X_transformed.shape[1]))
                self.classifier=LogisticRegressionCV()
                #self.classifier = RandomForestClassifier(min_impurity_decrease=0.05, max_features=None)

        start = time.time()
        self.classifier.fit(X_transformed, y)  # fit the classifier
        end = time.time()
        self.time_classifier = end-start
        
        return self

    def predict(self, X):

        check_is_fitted(self)  # make sure the classifier is fitted

        X = check_array(X)  # validate the shape of X

        # subsequence transform of X
        X_transformed = apply_kernels(X, self.kernels_)

        return self.classifier.predict(X_transformed)

    def predict_proba(self, X):
        check_is_fitted(self)  # make sure the classifier is fitted

        X = check_array(X)  # validate the shape of X

        # subsequence transform of X
        X_transformed = apply_kernels(X, self.kernels_)

        if isinstance(self.classifier, LinearClassifierMixin):
            return self.classifier._predict_proba_lr(X_transformed)
        return self.classifier.predict_proba(X_transformed)


if __name__ == "__main__":
    from sktime.datasets import load_UCR_UEA_dataset
    import time
    ds='MedicalImages' # Chosing a dataset from # Number of classes to consider

    X_train, y_train = load_UCR_UEA_dataset(name=ds, extract_path='data', split="train", return_type="numpy2d")
    X_test, y_test = load_UCR_UEA_dataset(name=ds, extract_path='data', split="test", return_type="numpy2d")
    #X_train = np.arange(10, dtype=np.float32).reshape((2, 5))
    #y_train = np.array([0, 1])

    #X_test = np.arange(10, dtype=np.float32).reshape((2, 5))
    #y_test = np.array([0, 1])
    
    # SAST
    start = time.time()
    sast = SAST(cand_length_list=np.arange(3, len(X_train)),
                nb_inst_per_class=1, classifier=RidgeClassifierCV())

    sast.fit(X_train, y_train)
    end = time.time()
    print('sast score :', sast.score(X_test, y_test))
    print('duration:', end-start)
    print('params:', sast.get_params())
    
 
    #print("X_train",X_train)
    #print("X_test",X_test)
    """
    start = time.time()
    random_state = None
    rsast_ridge = RSAST(n_random_points=5,nb_inst_per_class=5, sel_inst_wrepl=False,sel_randp_wrepl=True)
    rsast_ridge.fit(X_train, y_train)
    end = time.time()
    print('rsast score (sel_inst_wrepl=False,sel_randp_wrepl=True):', rsast_ridge.score(X_test, y_test))
    print('duration:', end-start)
    print('params:', rsast_ridge.get_params())

    start = time.time()
    random_state = None
    rsast_ridge = RSAST(n_random_points=5,nb_inst_per_class=5, sel_inst_wrepl=True,sel_randp_wrepl=True)
    rsast_ridge.fit(X_train, y_train)
    end = time.time()
    print('rsast score (sel_inst_wrepl=True,sel_randp_wrepl=True):', rsast_ridge.score(X_test, y_test))
    print('duration:', end-start)
    print('params:', rsast_ridge.get_params())

    start = time.time()
    random_state = None
    rsast_ridge = RSAST(n_random_points=5,nb_inst_per_class=5, sel_inst_wrepl=True, sel_randp_wrepl=False)
    rsast_ridge.fit(X_train, y_train)
    end = time.time()
    print('rsast score (sel_inst_wrepl=True,sel_randp_wrepl=False):', rsast_ridge.score(X_test, y_test))
    print('duration:', end-start)
    print('params:', rsast_ridge.get_params())

    start = time.time()
    random_state = None
    rsast_ridge = RSAST(n_random_points=5,nb_inst_per_class=5, sel_inst_wrepl=False, sel_randp_wrepl=False)
    rsast_ridge.fit(X_train, y_train)
    end = time.time()
    print('rsast score (sel_inst_wrepl=False,sel_randp_wrepl=False):', rsast_ridge.score(X_test, y_test))
    print('duration:', end-start)
    print('params:', rsast_ridge.get_params())
    
    start = time.time()
    random_state = None
    rsast_ridge = RSAST(half_instance=True, half_len=True, sel_inst_wrepl=False, sel_randp_wrepl=False)
    rsast_ridge.fit(X_train, y_train)
    end = time.time()
    print('rsast score (sel_inst_wrepl=False,sel_randp_wrepl=False) half instance half len:', rsast_ridge.score(X_test, y_test))
    print('duration:', end-start)
    print('params:', rsast_ridge.get_params())
    """
    start = time.time()
    random_state = None
    rsast_ridge = RSAST(n_random_points=10,nb_inst_per_class=10, len_method="all")
    rsast_ridge.fit(X_train, y_train)
    end = time.time()
    print('rsast score :', rsast_ridge.score(X_test, y_test))
    print('duration:', end-start)
    print('params:', rsast_ridge.get_params())
    

