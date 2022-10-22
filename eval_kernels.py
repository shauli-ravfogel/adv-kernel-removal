import os
import numpy as np
import numpy as np

from sklearn.linear_model import SGDClassifier, LinearRegression, Lasso, Ridge
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import seaborn as sn
import random
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.manifold import TSNE
import tqdm
import copy
from sklearn.svm import LinearSVC 

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import TruncatedSVD
import torch
from sklearn.linear_model import SGDClassifier

import sys
from pytorch_revgrad import RevGrad
from sklearn.svm import LinearSVC

import sklearn
from sklearn.linear_model import LogisticRegression
from extragradient import ExtraSGD
import random
import pickle
import matplotlib.pyplot as plt
from sklearn import neural_network
#from gensim.modenp.isnanls.keyedvectors import Word2VecKeyedVectors
#from gensim.models import KeyedVectors
from sklearn.svm import SVC
import scipy
import os
from sklearn.neural_network import MLPClassifier
import argparse
import run_kernels
from run_kernels import load_glove, load_synthetic, poly_kernel, rbf_kernel, sigmoid_kernel
from sklearn.metrics.pairwise import laplacian_kernel
from functools import *
import torch
from MKLpy.algorithms import AverageMKL, EasyMKL, KOMD

def laplace_kernel(X1,X2=None,gamma=None,degree=None,alpha=None):
    if X2 is None:
        X2 = X1    
    dists = torch.cdist(torch.tensor(X1),torch.tensor(X2)).detach().cpu().numpy()
    e = np.e
    return e**-(gamma*(dists))

def laplace_kernel_torch(X1,X2=None,gamma=None,degree=None,alpha=None):
    if X2 is None:
        X2 = X1    
    dists = torch.cdist(X1,X2) #scipy.spatial.distance_matrix(X1,X2)
    e = np.e
    return e**-(gamma*(dists))

def rbf_kernel(X1,X2=None,gamma=None,degree=None,alpha=None):
    if X2 is None:
        X2 = X1    
    dists = torch.cdist(torch.tensor(X1),torch.tensor(X2)).detach().cpu().numpy()
    e = np.e
    return e**-(gamma*(dists**2))


class AveragedKernel(object):
        def __init__(self, weights, kernel2params, kernels, numpy=True):
            
            kernels_lst = []
            for kernel_type in kernels.keys():
                for i,gamma in enumerate(kernel2params[kernel_type]["gammas"]):
                    for j,degree in enumerate(kernel2params[kernel_type]["degrees"]):
                        for k,alpha in enumerate(kernel2params[kernel_type]["alphas"]):
                        #print(kernel_type, gamma, degree, alpha)
                            kernels_lst.append({"func_np": kernels[kernel_type]["np"], "degree": degree, "gamma": gamma, "alpha": alpha})
                        
                        
            self.kernel_funcs = kernels_lst
            self.weights = weights
            self.numpy = numpy
    
        def __call__(self, x,y=None,gamma=None,alpha=None,degree=None):
        
            if y is None:
                y=x
            out = []
            for w,func_dict in zip(self.weights, self.kernel_funcs):
                
                if self.numpy:
                    func = func_dict["func_np"]
                else:
                    func = func_dict["func_torch"]
                    
                gamma = func_dict["gamma"]
                alpha = func_dict["alpha"]
                degree = func_dict["degree"]
                
                out.append(w*func(x,y,degree=degree,alpha=alpha,gamma=gamma))
            
            if self.numpy:
                return np.array(out).sum(axis=0)
            else:
                return torch.stack(out, dim=0).sum(dim=0)

gammas,degrees,alphas,alphas_sigmoid,gammas_sigmoid,gammas_laplace, gammas_rbf = run_kernels.gammas, run_kernels.degrees, run_kernels.alphas, run_kernels.alphas_sigmoid, run_kernels.gammas_sigmoid, run_kernels.gammas_laplace, run_kernels.gammas_rbf

kernels = run_kernels.kernels

kernel2params = run_kernels.kernel2params

def learn_multiple_kernels(kernels, kernel2params,X,y,numpy=False,equal_weighting=False):

    class AveragedKernel(object):
        def __init__(self, weights, kernel_funcs,numpy):
            self.kernel_funcs = kernel_funcs
            self.weights = weights
            self.numpy = numpy
    
        def __call__(self, x,y=None,gamma=None,alpha=None,degree=None):
        
            if y is None:
                y = x
            out = []
            for w,func_dict in zip(self.weights, self.kernel_funcs):
                
                if self.numpy:
                    func = func_dict["func_np"]
                else:
                    func = func_dict["func_torch"]
                    
                gamma = func_dict["gamma"]
                alpha = func_dict["alpha"]
                degree = func_dict["degree"]
                
                out.append(w*func(x,y,degree=degree,alpha=alpha,gamma=gamma))
            
            if self.numpy:
                return np.array(out).sum(axis=0)
            else:
                return torch.stack(out, dim=0).sum(dim=0)
        
            #return np.array([w*func["func"](x,y) for w,func in zip(self.weights, self.kernel_funcs)]).sum(axis=0)
    
    KLtr = []
    kernels_lst = []
    mlps = []

    for kernel_type in kernels.keys():
        if kernel_type == "EasyMKL" or kernel_type == "UniformMK":
            continue
        print(kernel_type)
        for i,gamma in enumerate(kernel2params[kernel_type]["gammas"]):
                for j,degree in enumerate(kernel2params[kernel_type]["degrees"]):
                    for k,alpha in enumerate(kernel2params[kernel_type]["alphas"]):
                        #print(kernel_type, gamma, degree, alpha)
                        KLtr.append(kernels[kernel_type]["np"](X, degree=degree, gamma=gamma, alpha=alpha))
                        kernels_lst.append({"func_np": kernels[kernel_type]["np"], "func_torch": kernels[kernel_type]["torch"], "degree": degree, "gamma": gamma, "alpha": alpha})

    if not equal_weighting:
        clf = EasyMKL().fit(KLtr,y)
        weights = clf.solution.weights.detach().cpu().numpy()
    else:
        weights = np.ones(len(kernels))/len(kernels)
    averaged_kernel_torch = AveragedKernel(weights, kernels_lst, numpy=False)
    averaged_kernel_np = AveragedKernel(weights, kernels_lst, numpy=True)
    return averaged_kernel_np


kernels2params_test = {"poly": {"degrees": [1,2,3], "alphas": [0.5], "gammas": [0.5]},
                      "rbf": {"gammas": [0.3], "degrees": [None], "alphas": [None]},
                      "laplace": {"gammas": [0.3], "degrees": [None], "alphas": [None]},
                      "sigmoid": {"gammas": [0.01], "degrees": [None], "alphas": [0.0]},
                      "linear": {"gammas": [None], "degrees": [None], "alphas": [None]}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An argparse example")
    parser.add_argument('--normalize', type=int, default=1, required=False)
    parser.add_argument('--run_id', type=int, default=-1, required=True)
    args = parser.parse_args()
    run_id = args.run_id
    os.makedirs("interim/glove{}/kernel/preimage/eval".format(run_id), exist_ok=True)
    os.makedirs("interim/glove{}/kernel/preimage/eval".format(run_id), exist_ok=True)
    
    X,Y,X_dev,Y_dev,X_test,Y_test = load_glove(normalize=args.normalize)
    print("majority dev: {}".format(max(Y_dev.mean(), 1-Y_dev.mean())))
    print("majority test: {}".format(max(Y_test.mean(), 1-Y_test.mean())))
    #X,Y,X_dev,Y_dev = load_synthetic(normalize=True, mode="poly")
    params2score = {}
    np.random.seed(run_id)
    random.seed(run_id)
    
    with open("interim/glove{}/kernel/multiple/multiple-kernels.pickle".format(run_id), "rb") as f:
        multiple_weights = pickle.load(f)
        easymkl_func = AveragedKernel(multiple_weights["EasyMKL"], kernel2params, kernels)
        unifommk_func = AveragedKernel(multiple_weights["UniformMK"], kernel2params, kernels)
        print("Done!")
    
    for filename in os.listdir("./interim/glove{}/kernel/preimage".format(run_id)):
        
        if not filename.endswith(".pickle"): continue
        if not "Z." in filename: continue
        if "UniformMK-RBF" in filename: continue
            
        params_str = (filename.rsplit('.', 1)[0]).split('.', 1)[1]
        params_dict = {kv.split("=")[0]:kv.split("=")[1] for kv in params_str.split("_")}

        gamma,degree,alpha = float(params_dict["gamma"]) if params_dict["gamma"]!="None" else 1, int(params_dict["degree"]) if params_dict["degree"]!="None" else 0, float(params_dict["alpha"]) if params_dict["alpha"]!="None" else 0
        with open("./interim/glove{}/kernel/preimage/Z.{}.pickle".format(run_id, params_str), "rb") as f:
            Z,error, mean_norm_normalized, post_proj_acc = pickle.load(f)
            #Z = Z / np.linalg.norm(Z, axis = 1, keepdims = True)
        with open("./interim/glove{}/kernel/preimage/Z_dev.{}.pickle".format(run_id, params_str), "rb") as f:
            Z_dev,error, mean_norm_normalized, post_proj_acc = pickle.load(f)
            #Z_dev = Z_dev / np.linalg.norm(Z_dev, axis = 1, keepdims = True)
        with open("./interim/glove{}/kernel/preimage/Z_test.{}.pickle".format(run_id, params_str), "rb") as f:
            Z_test,error, mean_norm_normalized, post_proj_acc = pickle.load(f)
            
        
        easymkl_func_best = learn_multiple_kernels(kernels, kernels2params_test, Z,Y)
        
        for kernel_type in ["UniformMK", "EasyMKL", "rbf", "poly","laplace","sigmoid","linear","mlp"]:
            
            if kernel_type == "mlp":
                model = MLPClassifier()
            else:            
                #if kernel_type == "UniformMK":
                #    model = SVC(kernel=unifommk_func.__call__)
                if kernel_type == "EasyMKL":
                    model = SVC(kernel=easymkl_func_best.__call__)
                elif kernel_type == "UniformMK":
                    model = SVC(kernel=unifommk_func.__call__) 
                elif kernel_type == "laplace":
                    model = SVC(kernel=partial(laplace_kernel, gamma=0.3)) #1.0/Z.shape[1]))
                elif kernel_type == "rbf":
                    model = SVC(kernel=partial(rbf_kernel, gamma=0.3)) #1.0/Z.shape[1]))
                elif kernel_type == "sigmoid":
                    model = SVC(kernel=kernel_type, gamma = 0.01)
                elif kernel_type == "poly":
                    model = SVC(kernel=kernel_type, degree=3, coef0 = 0.5, gamma = 0.5)
                else:      
                    model = SVC(kernel=kernel_type)
            print("Fitting...")
            model.fit(Z,Y)
            score = model.score(Z_test,Y_test)
            
            
            params_str = params_str.replace(".pickle", "_post-proj-acc={}.pickle".format(post_proj_acc))
            params_str = params_str.replace(".pickle", "_preimage-error={}.pickle".format(error/mean_norm_normalized))
            preimage_err_str, post_proj_acc_str = "{:.3f}".format(error*100/mean_norm_normalized),  "{:.3f}".format(100*post_proj_acc), 
            params_str = params_str.replace(".pickle", "_preimage-error={}.pickle".format(preimage_err_str))
            params_str = params_str.replace(".pickle", "post-proj-acc={}.pickle".format(post_proj_acc_str))
            params2score[params_str.replace("_unit-vecs=False","")+"_"+"adv-type={}".format(kernel_type)] = score
  
            score_same=None     
            if kernel_type == params_dict["kernel-type"]:# and kernel_type not in ["UniformMK"]:
                if kernel_type == "laplace":
                    model = SVC(kernel=partial(laplace_kernel, gamma=gamma))
                elif kernel_type == "rbf":
                    model = SVC(kernel=partial(rbf_kernel, gamma=gamma))
                elif kernel_type == "EasyMKL":
                    model = model = SVC(kernel=easymkl_func.__call__)
                elif kernel_type == "UniformMK":
                    model = model = SVC(kernel=unifommk_func.__call__)
                else:
                    print("using parameters", alpha, gamma, degree)
                    model = SVC(kernel=kernel_type, gamma=gamma, degree=degree, coef0=alpha)
                model.fit(Z,Y)
                score_same = model.score(Z_test,Y_test)
                params2score[params_str.replace("_unit-vecs=True","")+"_"+"adv-type={}_same".format(kernel_type)] = score_same
            print(params_str, kernel_type, score, score_same, post_proj_acc)
            print("Relative preimage error: {} %; post-proj-acc: {} %".format(preimage_err_str, post_proj_acc_str))
            print("-----------------------------------")
        print("=================================================")            
    with open("interim/glove{}/kernel/preimage/eval/scores_with_multiple2.pickle".format(run_id), "wb") as f:
            pickle.dump(params2score,f)
           
          