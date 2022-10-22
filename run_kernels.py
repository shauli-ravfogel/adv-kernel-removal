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
sys.path.append("MKLpy")

import inlp, inlp_dataset_handler, inlp_linear_model
from pytorch_revgrad import RevGrad
from sklearn.svm import LinearSVC

import sklearn
from sklearn.linear_model import LogisticRegression
import random
import pickle
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import neural_network
#from gensim.modenp.isnanls.keyedvectors import Word2VecKeyedVectors
#from gensim.models import KeyedVectors
from relaxed_inlp import solve_fantope_relaxation
from sklearn.svm import SVC
import scipy
import os
from MKLpy.metrics import pairwise
from MKLpy.algorithms import AverageMKL, EasyMKL, KOMD
import torch.nn.functional as F
import sys
import argparse

ALPHA = 0.01
TOL = 1e-4
ITER_NO_CHANGE = 900
LOSS = "log"


def load_glove(normalize):
    
    with open("glove-gender-data.pickle", "rb") as f:
        data_dict = pickle.load(f)
        X,Y,words_train = data_dict["train"]
        X_dev, Y_dev,words_dev = data_dict["dev"]
        X_test, Y_test, words_test = data_dict["test"]
        
        X,Y = X[Y > -1], Y[Y>-1]
        X_dev,Y_dev = X_dev[Y_dev > -1], Y_dev[Y_dev>-1]
        X_test, Y_test = X_test[Y_test > -1], Y_test[Y_test>-1]
        
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        X_dev = np.concatenate([X_dev, np.ones((X_dev.shape[0], 1))], axis=1)
        X_test = np.concatenate([X_test, np.ones((X_test.shape[0], 1))], axis=1)
        
        if normalize:
            #std = np.std(X, axis=0, keepdims=True)
            #std_dev = np.std(X_dev, axis=0, keepdims=True)
            
            #X = (X - np.mean(X, axis=0, keepdims=True))#/(std+1e-6)
            #X_dev = (X_dev - np.mean(X_dev, axis=0, keepdims=True))#/(std_dev+1e-6)
            
            X/=np.linalg.norm(X,axis=1,keepdims=True)
            X_dev/=np.linalg.norm(X_dev,axis=1,keepdims=True)
            X_test/=np.linalg.norm(X_test, axis=1, keepdims=True)
            
    return X,Y,X_dev,Y_dev, X_test, Y_test



def load_biasbios(normalize,pca_dim=300):

    data = []

    pca = PCA(n_components=pca_dim, random_state=0)
    for mode in ["train", "dev", "test"]:
        X = np.load("{}_cls.npy".format(mode))
        with open("{}.pickle".format(mode), "rb") as f:
            bios_data = pickle.load(f)
            Y = np.array([1 if d["g"]=="f" else 0 for d in bios_data])
            X,Y = sklearn.utils.shuffle(X,Y, random_state=0)
            if mode == "train":
                pca.fit(X)
            X = pca.transform(X)
            if normalize:
                X/=np.linalg.norm(X,axis=1,keepdims=True)
            Y[Y<1e-6] = -1
            Y[Y<1e-6] = 0
            data.extend([X,Y])
        
    return data


def load_synthetic(normalize, mode):
    
    if mode == "poly":
        
        fname = "datasets/synthetic/kernels/kernel_type=poly_gamma=0.2_degree=3_alpha=0.5.pickle"
        
    elif mode == "rbf":
        
        fname = "datasets/synthetic/kernels/kernel_type=poly_gamma=0.3_degree=3_alpha=0.5.pickle"
        
    elif mode == "laplace":
        
        fname = "datasets/synthetic/kernels/kernel_type=poly_gamma=0.3_degree=3_alpha=0.5.pickle"
        
    with open(fname, "rb") as f:
        X,Y,X_dev,Y_dev,X_test, Y_test = pickle.load(f)

        if normalize:
            X/=np.linalg.norm(X,axis=1,keepdims=True)
            X_dev/=np.linalg.norm(X_dev,axis=1,keepdims=True)
            
    return X,Y,X_dev,Y_dev

def project_2d(X,y,method="pca",M=750,title="",mode="glove"):
    
    if method == "pca":
        model = PCA(n_components=2)
    else:
        model = TSNE(n_components=2)
    
    projected = model.fit_transform(X[:M])
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Serif'
    sn.set_style("darkgrid")
    ax = plt.axes()
    plot = sn.scatterplot(projected[:, 0], projected[:, 1], hue =y[:M])
    plt.legend(fontsize=19)
    plt.savefig("plots/{}/kernels/tsne-pca/{}.method={}.png".format(mode,title,method), dpi = 600)
    plt.clf()



def calc_nystrom_old(X,d, kernel_func,alpha,gamma,degree, eps = 1e-7): 
    """ 
    https://pdfs.semanticscholar.org/5a80/7f0343c01040ac20740a405827982f06640d.pdf
    https://stats.stackexchange.com/questions/261149/nystroem-method-for-kernel-approximation
    Args:
        X: input matrix, NxD
        degree: number of nystrom landmarks
        kernel_func: a function that maps (X1,X2) to a similarity matrix K where K[i][j] is sim(X1[i], X2[j])
    Return:
        A matix X_tilde, N x degree, s.t. K is approximated by X_tilde@X_tilde.T
    """
    
    X1,X2 = X[:d], X[d:] # X1 is degree x D, X2 is (N-degree) x D
    W = kernel_func(X1,X1,gamma=gamma,degree=degree,alpha=alpha) # W is degree x degree
    X2X1 = kernel_func(X2,X1,gamma=gamma,degree=degree,alpha=alpha) # N-degree x degree
    C = np.concatenate([W,X2X1], axis=0) # C is N x degree
   
    eigvals,U = np.linalg.eigh(W)
    eigvals[eigvals<eps] = eps
    
    S_sqrt_inv = np.diag(eigvals**(-0.5))
    X_tilde = C@U@S_sqrt_inv
        
    return X_tilde, U@S_sqrt_inv, X1

def calc_nystrom(X,d, kernel_func,alpha,gamma,degree, eps = 1e-10): 
    """ 
    https://pdfs.semanticscholar.org/5a80/7f0343c01040ac20740a405827982f06640d.pdf
    https://stats.stackexchange.com/questions/261149/nystroem-method-for-kernel-approximation
    Args:
        X: input matrix, NxD
        degree: number of nystrom landmarks
        kernel_func: a function that maps (X1,X2) to a similarity matrix K where K[i][j] is sim(X1[i], X2[j])
    Return:
        A matix X_tilde, N x degree, s.t. K is approximated by X_tilde@X_tilde.T
    """
    
    X1,X2 = X[:d], X[d:] # X1 is degree x D, X2 is (N-degree) x D
    W = kernel_func(X1,X1,gamma=gamma,degree=degree,alpha=alpha) # W is degree x degree
    X2X1 = kernel_func(X2,X1,gamma=gamma,degree=degree,alpha=alpha) # N-degree x degree
    C = np.concatenate([W,X2X1], axis=0) # C is N x degree
   
    eigvals,U = np.linalg.eigh(W)
    eigvals[eigvals<eps]=eps
    S_sqrt_inv = np.diag(eigvals**(-0.5))
    X_tilde = C@U@S_sqrt_inv
    
    V = U@S_sqrt_inv
    
    return X_tilde, V, X1


def poly_kernel(X1,X2=None,gamma=None,degree=None,alpha=None):
    if X2 is None:
        X2 = X1
    if gamma is None:
        gamma = 1.0
    return (gamma * X1@X2.T + alpha)**degree
    
def linear_kernel(X1,X2=None,gamma=None,degree=None,alpha=None): 
    if X2 is None:
        X2 = X1    
    return X1@X2.T

def sigmoid_kernel(X1,X2=None,gamma=None,degree=None,alpha=None): 
    if X2 is None:
        X2 = X1
    return np.tanh(gamma*X1@X2.T + alpha)
    
def sigmoid_kernel_torch(X1,X2=None,gamma=None,degree=None,alpha=None): 
    if X2 is None:
        X2 = X1     
    return torch.tanh(gamma*X1@X2.T + alpha)
    
    
def rbf_kernel(X1,X2=None,gamma=None,degree=None,alpha=None):
    if X2 is None:
        X2 = X1    
    dists = torch.cdist(torch.tensor(X1),torch.tensor(X2)).detach().cpu().numpy()
    e = np.e
    return e**-(gamma*(dists**2))

def rbf_kernel_torch(X1,X2=None,gamma=None,degree=None,alpha=None):
    if X2 is None:
        X2 = X1    
    dists = torch.cdist(X1,X2) #scipy.spatial.distance_matrix(X1,X2)
    e = np.e
    return e**-(gamma*(dists**2))

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


class MLP(torch.nn.Module):
    def __init__(self, d_in, d_out):
        
        super(MLP, self).__init__()  # just run the init of parent class (nn.Module)
        self.lin1 = torch.nn.Linear(d_in, 512)
        self.lin2 = torch.nn.Linear(512,512)
        self.lin3 = torch.nn.Linear(512,d_out)
        self.lin4 = torch.nn.Linear(d_in, d_in)


        self.BatchNorm1d_1 = torch.nn.BatchNorm1d(512)
        self.BatchNorm1d_2 = torch.nn.BatchNorm1d(512)
        self.BatchNorm1d_3 = torch.nn.BatchNorm1d(301)
        self.Dropout1d = torch.nn.Dropout(0.1)
        
        self.W = torch.nn.Linear(d_out, d_out)
        
    def forward(self, x):        
        h = F.relu(self.lin1(x))
        h = self.BatchNorm1d_1(self.Dropout1d(h))
                
        h = F.relu(self.lin2(h))
        o = self.lin3(h) + x
        return o


def get_triplet_loss(z_pred, z_true, temp=10):
    
    dists = torch.cdist(z_pred, z_true)
    print(dists)
    num = torch.exp((1/temp)*torch.diag(dists)**2 )
    denum = torch.sum(torch.exp((1/temp)*dists**2), dim = 1)
    loss = torch.log(num/(denum + 1e-9))
    return loss.mean()
    

def get_l2_loss(z_pred, z_true, P_row):
    
    loss_rec = (torch.norm(z_pred - z_true, dim = 1)**2).mean()
    loss_rowspace = (torch.norm(z_pred@P_row, dim = 1)**2).mean()
    return 0.5 * (loss_rec + 1*loss_rowspace)
    
def calc_preimage_nystrom_mse(X, phi_X_debiased, X_dev, phi_X_dev_debiased, X1, S, kernel_func, gamma,degree,alpha,P,device="cpu"):
    
    d_in, d_out = X.shape[1], X.shape[1]
    X_torch = torch.tensor(X).float().to(device)
    X_dev_torch = torch.tensor(X_dev).float().to(device)
    
    
    diff_torch = 1e-3*torch.randn_like(X_torch).float().to(device)
    
    X1_torch = torch.tensor(X1).float().to(device)
    S_torch = torch.tensor(S).float().to(device)
    phi_X_debiased_torch = torch.tensor(phi_X_debiased).float().to(device)
    phi_X_dev_debiased_torch = torch.tensor(phi_X_dev_debiased).float().to(device)
    phi_X_torch = kernel_func(X_torch, X1_torch,gamma=gamma,degree=degree,alpha=alpha)@S_torch
    phi_X_dev_torch = kernel_func(X_dev_torch, X1_torch,gamma=gamma,degree=degree,alpha=alpha)@S_torch
    
    X_torch.requires_grad=False
    diff_torch.requires_grad=True
    mlp = (MLP(d_in, d_out)).to(device)
    optimizer = torch.optim.Adam(mlp.parameters())
    
    loss_fn = torch.nn.MSELoss()
    P_rowspace = (torch.eye(P.shape[0]).float() - torch.tensor(P).float()).to(device)
    
    import copy
    best_mlp,best_score = copy.deepcopy(mlp), 10000
    
    mean_loss = []
    mean_loss_dev = []
    N_TOTAL=20000
    
    assert X_torch.shape[0] == phi_X_debiased_torch.shape[0]
    assert X_dev_torch.shape[0] == phi_X_dev_debiased_torch.shape[0]
    
    for i in range(N_TOTAL):
        
        optimizer.zero_grad()
        perm = torch.randperm(X_torch.shape[0], device = device)
        idx =  perm[:128]
        
        inp = X_torch[idx]
        
        phi_Z = kernel_func(mlp(inp), X1_torch,gamma=gamma,degree=degree,alpha=alpha)@S_torch
        y = phi_X_debiased_torch[idx]
        
        if i % 1000 == 0:
            with torch.no_grad():
                mlp.eval()
                y_dev = phi_X_dev_debiased_torch
                inp = X_dev_torch
                phi_Z_dev = kernel_func(mlp(inp), X1_torch,gamma=gamma,degree=degree,alpha=alpha)@S_torch
                
                loss_dev = get_l2_loss(phi_Z_dev, y_dev, P_rowspace)
                mean_loss_dev.append(loss_dev.detach().cpu().numpy().item())

                print("{}/{}; dev loss: {}; train loss: {}".format(i, N_TOTAL, mean_loss_dev[-1], np.mean(mean_loss) ))
                #exit()
                mlp.train()
                if loss_dev.detach().cpu().numpy().item() < best_score:
                    best_score = loss_dev.detach().cpu().numpy().item()
                    best_mlp = copy.deepcopy(mlp)

        loss = get_l2_loss(phi_Z, y, P_rowspace)
        mean_loss.append(loss.detach().cpu().numpy().item())
        
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            mean_loss = []
    
    best_mlp = best_mlp.eval()
    with torch.no_grad():
           best_mlp.eval()
           y_dev = phi_X_dev_debiased_torch
           inp = X_dev_torch
           phi_Z_dev = kernel_func(best_mlp(inp), X1_torch,gamma=gamma,degree=degree,alpha=alpha)@S_torch
           L2_loss_dev = get_l2_loss(phi_Z_dev, y_dev, P_rowspace)
           L2_loss_dev = L2_loss_dev.detach().cpu().numpy().item()#

    error = np.sqrt(L2_loss_dev+1e-10)
    mean_norm_normalized = (phi_X_dev_debiased_torch.norm(dim=1)).mean().detach().cpu().numpy()
    
    inp = X_torch
    inp_dev = X_dev_torch
    with torch.no_grad():
        Z = best_mlp(inp).detach().cpu().numpy()
        Z_dev = best_mlp(inp_dev).detach().cpu().numpy()
        
    return Z,Z_dev,best_mlp, error, mean_norm_normalized


def get_svd(X):
    
    D,U = np.linalg.eigh(X)
    return U,D

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
            
    KLtr = []
    kernels_lst = []

    for kernel_type in kernels.keys():
        if kernel_type == "EasyMKL" or kernel_type == "UniformMK":
            continue
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
    return averaged_kernel_torch, averaged_kernel_np, weights, kernels_lst
    

    
gammas = [0.05, 0.1, 0.15]
alphas = [0.8,1,1.2]
gammas_sigmoid = [0.005, 0.003]
gammas_rbf = [0.1, 0.15, 0.2]
gammas_laplace=gammas_rbf
alphas_sigmoid = [0, 0.01]
degrees = [2,3]
do_normalize = [False]

kernels = {"poly": {"torch": poly_kernel, "np": poly_kernel}, "rbf": {"torch": rbf_kernel_torch, "np": rbf_kernel},
           "laplace": {"torch": laplace_kernel_torch, "np": laplace_kernel}, "linear": {"torch": linear_kernel, "np": linear_kernel}, "sigmoid": {"torch": sigmoid_kernel_torch, "np": sigmoid_kernel}} 

kernel2params = {"poly": {"gammas": gammas, "degrees": degrees, "alphas": alphas},
                 "rbf": {"gammas": gammas_rbf, "degrees": [None], "alphas": [None]},
                 "laplace": {"gammas": gammas_laplace, "degrees": [None], "alphas": [None]},
                 "linear": {"gammas": [None], "degrees": [None], "alphas": [None]},
                 "sigmoid": {"gammas": gammas_sigmoid, "degrees": [None], "alphas": alphas_sigmoid}}


if __name__ == "__main__":

 os.makedirs("datasets/synthetic/kernels/kernels-heatmaps", exist_ok=True)
 os.makedirs("datasets/glove2/kernels/kernels-heatmaps", exist_ok=True)

 os.makedirs("plots/glove2/kernels/kernels-heatmaps", exist_ok = True)
 os.makedirs("plots/glove2/kernels/eigvals", exist_ok=True)
 os.makedirs("plots/glove2/kernels/tsne-pca", exist_ok=True)
 os.makedirs("interim/glove2/kernel/projected", exist_ok=True)
 os.makedirs("interim/glove2/kernel/nystrom", exist_ok=True)
 os.makedirs("interim/glove2/kernel/preimage", exist_ok=True)
    
 os.makedirs("datasets/glove3/kernels/kernels-heatmaps", exist_ok=True)
 os.makedirs("plots/glove3/kernels/kernels-heatmaps", exist_ok = True)
 os.makedirs("plots/glove3/kernels/eigvals", exist_ok=True)
 os.makedirs("plots/glove3/kernels/tsne-pca", exist_ok=True)
 os.makedirs("interim/glove3/kernel/projected", exist_ok=True)
 os.makedirs("interim/glove3/kernel/nystrom", exist_ok=True)
 os.makedirs("interim/glove3/kernel/preimage", exist_ok=True)
    
 os.makedirs("plots/poly/kernels/kernels-heatmaps", exist_ok=True)
 os.makedirs("plots/poly/kernels/eigvals", exist_ok=True)
 os.makedirs("plots/poly/kernels/tsne-pca", exist_ok=True)
 os.makedirs("interim/poly/kernel/projected", exist_ok=True)
 os.makedirs("interim/poly/kernel/nystrom", exist_ok=True)
 os.makedirs("interim/poly/kernel/preimage", exist_ok=True)

 os.makedirs("plots/rbf/kernels/kernels-heatmaps", exist_ok=True)
 os.makedirs("plots/rbf/kernels/eigvals", exist_ok=True)
 os.makedirs("plots/rbf/kernels/tsne-pca", exist_ok=True)
 os.makedirs("interim/rbf/kernel/projected", exist_ok=True)
 os.makedirs("interim/rbf/kernel/nystrom", exist_ok=True)
 os.makedirs("interim/rbf/kernel/preimage", exist_ok=True)
    
    
 mode = sys.argv[1].replace("--", "")
 parser = argparse.ArgumentParser(description="An argparse example")
 parser.add_argument('--normalize', type=int, default=1, required=False)
 parser.add_argument('--run_id', type=int, default=-1, required=True)
 parser.add_argument('--device', type=int, default=-1, required=True)
 parser.add_argument('--mode', type=str, default="glove", required=True)
 parser.add_argument('--kernel_type', type=str, default=None, required=False)

                 
 args = parser.parse_args()
 run_id = args.run_id
 mode = args.mode  
 if args.kernel_type is not None:
    kernels = {args.kernel_type: kernels[args.kernel_type]}

 if mode == "glove":
     print("Loading glove")
     X,Y,X_dev,Y_dev,X_test, Y_test = load_glove(normalize=True)
 else:
     print("Loading bios")
     X,Y,X_dev,Y_dev, X_test, Y_test= load_biasbios(normalize=True)

 NN = 50000
    
 random.seed(0)
 np.random.seed(0)
 X,Y = X[:NN], Y[:NN]
 
 if len(kernels) > 1:
    K_multiple_torch, K_multiple_np, _, _ = learn_multiple_kernels(kernels, kernel2params, X, Y, equal_weighting=False)
    kernels["EasyMKL"] = {"torch": K_multiple_torch, "np": K_multiple_np} 
    kernel2params["EasyMKL"] = {"gammas": [None], "degrees": [None], "alphas": [None]}
    K_multiple_torch_eq, K_multiple_np_eq, _, _ = learn_multiple_kernels(kernels, kernel2params, X, Y, equal_weighting=True)
    kernels["UniformMK"] = {"torch": K_multiple_torch_eq, "np": K_multiple_np_eq} 
    kernel2params["UniformMK"] = {"gammas": [None], "degrees": [None], "alphas": [None]}
    K_multiple_torch_eq_rbf, K_multiple_np_eq_rbf, _, _ = learn_multiple_kernels({"rbf": kernels["rbf"]}, kernel2params, X, Y, equal_weighting=False)
    kernels["UniformMK-RBF"] = {"torch": K_multiple_torch_eq_rbf, "np": K_multiple_np_eq_rbf} 
    kernel2params["UniformMK-RBF"] = {"gammas": [None], "degrees": [None], "alphas": [None]}
    with open("interim/{}{}/kernel/multiple/multiple-kernels.pickle".format(mode,run_id), "wb") as f:
                pickle.dump({"EasyMKL": copy.deepcopy(K_multiple_np.weights), "UniformMK": copy.deepcopy(K_multiple_np_eq.weights),
                            "UniformMK-RBF": copy.deepcopy(K_multiple_np_eq_rbf.weights)}, f)
 NUM_RNS = 10
 for i in range(4,20):
         os.makedirs("datasets/glove{}/kernels/kernels-heatmaps".format(i), exist_ok=True)
         os.makedirs("plots/glove{}/kernels/kernels-heatmaps".format(i), exist_ok = True)
         os.makedirs("plots/glove{}/kernels/eigvals".format(i), exist_ok=True)
         os.makedirs("plots/glove{}/kernels/tsne-pca".format(i), exist_ok=True)
         os.makedirs("interim/glove{}/kernel/projected".format(i), exist_ok=True)
         os.makedirs("interim/glove{}/kernel/nystrom".format(i), exist_ok=True)
         os.makedirs("interim/glove{}/kernel/preimage".format(i), exist_ok=True)
         os.makedirs("interim/glove{}/kernel/multiple".format(i), exist_ok=True)
         os.makedirs("interim/bios{}/kernel/projected".format(i), exist_ok=True)
         os.makedirs("interim/bios{}/kernel/nystrom".format(i), exist_ok=True)
         os.makedirs("interim/bios{}/kernel/preimage".format(i), exist_ok=True)
         os.makedirs("interim/bios{}/kernel/multiple".format(i), exist_ok=True)

 
 
 random.seed(run_id)
 np.random.seed(run_id)
                 
 for d in [1024]:
    for kernel_type in kernels.keys():
        for normalize in [args.normalize]:
            if mode == "glove":
                X,Y,X_dev,Y_dev,X_test, Y_test = load_glove(normalize=normalize)
            elif mode == "bios":
                X,Y,X_dev,Y_dev,X_test, Y_test = load_biasbios(normalize=normalize)
            elif mode == "synthetic":
                X,Y,X_dev,Y_dev = load_synthetic(normalize=True)
            X,Y=X[:NN],Y[:NN]
            
            for i,gamma in enumerate(kernel2params[kernel_type]["gammas"]):
                for j,degree in enumerate(kernel2params[kernel_type]["degrees"]):
                    for k,alpha in enumerate(kernel2params[kernel_type]["alphas"]):
                        
                        # calculate kernel approximation

                        X_kernel, S, X1 = calc_nystrom(X.copy(),d=d,kernel_func=kernels[kernel_type]["np"], alpha=alpha,gamma=gamma,degree=degree)
                        X_dev_kernel = kernels[kernel_type]["np"](X_dev, X1, gamma=gamma, degree=degree, alpha=alpha)@S
                        X_test_kernel = kernels[kernel_type]["np"](X_test, X1, gamma=gamma, degree=degree, alpha=alpha)@S
                 
                        print(kernel_type, gamma, degree, alpha)
                        if np.isnan(np.sum(X_kernel)): continue
                        
                        params_str = "kernel-type={}_d={}_gamma={}_degree={}_alpha={}".format(kernel_type, d, str(gamma), str(degree), str(alpha))
                        
                        # run adversarial game
                        
                        RANK=1

                        ws, advs, best_adv, best_score = solve_fantope_relaxation(X_kernel, Y ,d=RANK,init_beta=None,device=args.device, out_iters=35000, in_iters_adv=1, in_iters_clf=1, batch_size = 128, epsilon=0.025, noise_std=1e-3, lr=0.01, weight_decay=1e-5,
                        momentum=0.0)
                        
                        
                        U,D = get_svd(best_adv)
                        U = U.T
                        W = U[-RANK:] 
                        P = np.eye(X_kernel.shape[1]) - W.T@W
                        with open("interim/{}{}/kernel/P.{}.pickle".format(mode, run_id, params_str), "wb") as f:
                            pickle.dump(P, f)
                 
                        X_kernel_proj = X_kernel@P
                        X_dev_kernel_proj = X_dev_kernel@P
                        X_test_kernel_proj = X_test_kernel@P
                 
                        with open("interim/{}{}/kernel/projected/X.proj.{}.pickle".format(mode,run_id,params_str), "wb") as f:                   
                            pickle.dump(X_kernel_proj, f)
                        with open("interim/{}{}/kernel/projected/X_dev.proj.{}.pickle".format(mode,run_id,params_str), "wb") as f:               
                            pickle.dump(X_dev_kernel_proj, f)
                        with open("interim/{}{}/kernel/projected/X_test.proj.{}.pickle".format(mode,run_id,params_str), "wb") as f:               
                            pickle.dump(X_test_kernel_proj, f)
                 
                        with open("interim/{}{}/kernel/nystrom/X.{}.pickle".format(mode,run_id,params_str), "wb") as f:                       
                            pickle.dump(X_kernel, f)
                        with open("interim/{}{}/kernel/nystrom/X_dev.{}.pickle".format(mode,run_id,params_str), "wb") as f:                       
                            pickle.dump(X_dev_kernel, f)
                        with open("interim/{}{}/kernel/nystrom/X_test.{}.pickle".format(mode,run_id,params_str), "wb") as f:                       
                            pickle.dump(X_test_kernel, f)    
                        # run preimage
                        mlps = []
                        for random_try in range(1):
                            best_mlp, best_Z, best_Z_dev, best_error, best_norm = None, None, None, 10000, None
                            
                            for q in range(1):
                               
                                Z,Z_dev,mlp, error, mean_norm_normalized = calc_preimage_nystrom_mse(X,X_kernel_proj, X_dev, X_dev_kernel_proj, X1,S,kernels[kernel_type]["torch"], gamma, degree, alpha,P, device=args.device)
                                if error < best_error:
                                   best_mlp, best_Z, best_Z_dev, best_error, best_mean_norm_normalized = copy.deepcopy(mlp), copy.deepcopy(Z), copy.deepcopy(Z_dev), error, mean_norm_normalized
                            
                            mlp = best_mlp
                            Z = best_Z
                            Z_dev = best_Z_dev
                            error = best_error
                            mean_norm_normalized = best_mean_norm_normalized
                            with torch.no_grad():
                                Z_test = best_mlp(torch.tensor(X_test).to(args.device).float()).detach().cpu().numpy()
                 
                            print("error: {}; mean_norm: {}; relative error: {}".format(error,mean_norm_normalized, error*100/mean_norm_normalized))
                            with open("interim/{}{}/kernel/preimage/MLP.{}.pickle".format(mode,run_id, params_str), "wb") as f:
                                pickle.dump((mlp.cpu(), error, mean_norm_normalized), f)
                        
                            with open("interim/{}{}/kernel/preimage/Z.{}.pickle".format(mode,run_id, params_str), "wb") as f:                       
                                pickle.dump((Z,error, mean_norm_normalized, best_score), f)
                                #Z_dev = calc_preimage_nystrom_mse(X_dev,X_dev_kernel_proj,X1,S,kernels[kernel_type]["torch"], gamma, degree, alpha,P)
                        
                                with open("interim/{}{}/kernel/preimage/Z_dev.{}.pickle".format(mode,run_id,params_str), "wb") as f:                       
                                    pickle.dump((Z_dev, error, mean_norm_normalized, best_score), f)
                                with open("interim/{}{}/kernel/preimage/Z_test.{}.pickle".format(mode,run_id,params_str), "wb") as f:                       
                                    pickle.dump((Z_test, error, mean_norm_normalized, best_score), f)
                 
                        
                    




