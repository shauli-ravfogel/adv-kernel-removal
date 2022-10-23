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


from sklearn.svm import LinearSVC

import sklearn
from sklearn.linear_model import LogisticRegression
import random
import pickle
from sklearn import cluster
from sklearn import neural_network
import time

ALPHA = 0.5*1e-4
TOL = 0.25*1e-3
ITER_NO_CHANGE = 25
LOSS = "log"



def evaluate(w, X,Y, verbose=True):

    w = w/np.linalg.norm(w)
    P = np.eye(X.shape[1]) - np.outer(w,w)
    
    if verbose:
        print("Is Projection: {}; is symmetric: {}".format(np.allclose(P@P-P, 0.), np.allclose(P-P.T,0.)))
    scores = []
    losses = []
    
    for i in range(2):
        svm = SGDClassifier(loss=LOSS, fit_intercept=True,  max_iter=120000, tol = TOL,n_iter_no_change=ITER_NO_CHANGE,
                           n_jobs=64,alpha=ALPHA)
        svm.fit(X@P,Y)
        scores.append(svm.score(X@P,Y))
        if LOSS == "hinge":
            losses.append(sklearn.metrics.hinge_loss(Y, svm.predict(X@P)))
        elif LOSS == "log":
            losses.append(sklearn.metrics.log_loss(Y, svm.predict(X@P)))
    
    if verbose:
        print("\tAccuracy after projection:")
        print("\t\tMean: {}; Median: {}; Min: {}; Max: {}; STD: {}".format(np.mean(scores), np.median(scores), np.min(scores), 
                                                         np.max(scores), np.std(scores)))
        print("\tLoss after projection:")
        print("\t\tMean: {}; Median: {}; Min: {}; Max: {}; STD: {}".format(np.mean(losses), np.median(losses), np.min(losses), 
                                                         np.max(losses), np.std(losses)))
    return np.mean(losses), np.mean(scores)
    
def symmetric(X):
    X.data = 0.5 * (X.data + X.data.T)
    return X
    
def get_svd(X):
    
    D,U = np.linalg.eigh(X)
    return U,D
            
def get_score(X,Y, adv, rank):
    
    U,D = get_svd(adv)
    U = U.T
    W = U[-rank:] 
    P = np.eye(X.shape[1]) - W.T@W 
    svm = SGDClassifier(loss=LOSS, fit_intercept=True,  max_iter=40000, tol = TOL,n_iter_no_change=ITER_NO_CHANGE,
                           n_jobs=65,alpha=ALPHA)
    svm.fit(X[:]@P,Y[:])
    return svm.score(X[:]@P,Y[:])
    

def solve_constraint(lambdas, d=1):
    
    def f(theta):
        return_val = np.sum(np.minimum(np.maximum(lambdas - theta, 0), 1)) - d
        return return_val

    theta_min, theta_max = max(lambdas), min(lambdas) - 1
    assert f(theta_min)*f(theta_max)<0
        
    mid = (theta_min + theta_max)/2
    tol = 1e-4
    iters = 0
    
    while iters<25:
        
        mid = (theta_min + theta_max)/2
        
        if f(mid)*f(theta_min) > 0:
            
            theta_min = mid
        else:
            theta_max = mid
        iters += 1
        
    lambdas_plus = np.minimum(np.maximum(lambdas - mid, 0), 1)
    #if (theta_min-theta_max)**2 > tol:
    #    print("didn't converge", (theta_min-theta_max)**2)
    return lambdas_plus
    

def solve_fantope_relaxation_fr(X,Y,d=1,init_beta=None,device="cpu", out_iters=50000, in_iters_adv=1,
                             in_iters_clf = 1, epsilon = 0.001, batch_size = 16, replay = False):

    def get_loss_fn(X,Y,alpha,beta, bce_loss_fn, optimize_beta=False):
        I = torch.eye(X.shape[1]).to(device)
        bce = bce_loss_fn(alpha(X@(I - beta)).squeeze(), Y) + ALPHA * alpha.weight.data.norm()**2 
        if optimize_beta:
            bce = -bce
            
        return bce

    def solve_linear_minimization(grad, iters=15):

         s = torch.randn_like(grad)
         s.requires_grad = True
         optimizer = torch.optim.SGD([s], lr = 1e-2, momentum = 0.8)
         for i in range(iters):
                optimizer.zero_grad()
                loss = (grad*s).sum()
                loss.backward()
                optimizer.step()
           
                # project
            
                D,U = torch.linalg.eigh(symmetric(s))
                D = D.detach().cpu().numpy()
                D_plus_diag = solve_constraint(D,d=d)
                D = torch.tensor(np.diag(D_plus_diag).real).float().to(device)
                s.data = U@D@U.T
         return s
    
    
    # init models
    X_torch = torch.tensor(X).float().to(device)
    Y_torch = torch.tensor(Y).float().to(device)
    
    w = torch.nn.Linear(X.shape[1], 1).to(device) 
    A = torch.randn(d, X.shape[1]).to(device)
    #A = A / A.norm(dim=1, keepdim=True)
    beta = 1e-2*torch.randn(X.shape[1], X.shape[1]).to(device) #(A.T@A).to(device)
    print('inited')
    
    w.requires_grad = True
    beta.requires_grad = True
    
    # optimizers
    
    optimizer_w = ExtraSGD(w.parameters(), lr=0.1, momentum=0.9) #torch.optim.SGD(w.parameters(), lr = 1e-3, momentum = 0.6)
    optimizer_adv = ExtraSGD([beta], lr=0.1, momentum=0.9) #torch.optim.SGD([beta],  lr = 1e-3, momentum = 0.6)

    # loss, recordings
    
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    ws = [w.weight.data.detach().cpu().numpy().copy()]
    advs = [beta.detach().cpu().numpy().copy()]
    
    grads_beta = []
    grads_alpha = []
    best_adv, best_score = None, 1
    count_examples=0.
    best_adv = beta.detach().cpu().numpy().copy()
    print("starting...")
    for i in range(out_iters):
        
        gamma = 2/(2+i)
        
        for j in range(in_iters_adv):
            print(j)
            beta = symmetric(beta)
            optimizer_adv.zero_grad()
            
            idx = np.arange(0,X_torch.shape[0])
            np.random.shuffle(idx)
            X_batch, Y_batch = X_torch[idx[:batch_size]], Y_torch[idx[:batch_size]]
            
            loss_adv = get_loss_fn(X_batch, Y_batch, w, symmetric(beta), bce_loss_fn, optimize_beta = True)
            loss_adv.backward()
            optimizer_adv.step()
            grads_beta.append(beta.grad.detach().cpu().norm().numpy())
            
            s = solve_linear_minimization(beta.grad.detach().cpu())
            if np.random.rand() < 1e-2: print((beta.data-s).norm())
            beta.data += 0.1*s
            advs.append(beta.detach().cpu().numpy().copy())
            
        for j in range(in_iters_clf):
            
            optimizer_w.zero_grad()
            idx = np.arange(0,X_torch.shape[0])
            np.random.shuffle(idx)
            X_batch, Y_batch = X_torch[idx[:batch_size]], Y_torch[idx[:batch_size]]
            
            loss_w = get_loss_fn(X_batch, Y_batch, w, symmetric(beta), bce_loss_fn, optimize_beta = False)
            loss_w.backward()
            grads_alpha.append(w.weight.grad.detach().cpu().norm().numpy())
            optimizer_w.step()
            #w.weight.data = (1-gamma) * w.weight.data + gamma*(-w.weight.grad)
            #w.bias.data = (1-gamma) * w.bias.data + gamma*(-w.bias.grad)

        count_examples += batch_size    
            
        if i % 50 == 0:
            diffs_ws = np.mean([np.linalg.norm(ws[i]-ws[i-1]) for i in range(1,len(ws))])
            diffs_advs = np.mean([np.linalg.norm(advs[i]-advs[i-1]) for i in range(1,len(advs))])
            
            ws = ws[-1000:]
            advs = advs[-250:]
            
            print("{}/{}".format(i, out_iters))
  
            score = get_score(X,Y, advs[-1], d)
            print("Accuracy after projection:{}".format(score))
            if score < best_score:
                best_adv, best_score = symmetric(beta).detach().cpu().numpy().copy(), score
            grads_alpha = []
            grads_beta = []
            
            if i > 5 and np.abs(best_score - max(Y.mean(),1-Y.mean())) < epsilon:
                advs.append(symmetric(beta).detach().cpu().numpy().copy())
                return ws, advs, best_adv
    print("=============================")
    print("Best score is: {}".format(best_score))
    advs.append(symmetric(beta).detach().cpu().numpy().copy())
    return ws, advs, best_adv

            
def solve_fantope_relaxation(X,Y,d=1,init_beta=None,device="cpu", out_iters=50000, in_iters_adv=1,
                             in_iters_clf = 1, epsilon = 0.0015, batch_size = 128, replay = False, verbose=False, lr = 0.15,
                            noise_std=0.0, input_dropout=0.0, momentum=0.0, weight_decay=0.0, project_freq=1,evalaute_every=500,nuclear_norm_weight=0.0,learn_proj=False,use_adam=False):

    print("Dataset size: {}; device: {}; epsilon: {}; noise-std: {}, lr: {}; average feature size: {}; dropout: {}".format(X.shape, device, epsilon, noise_std, lr, np.mean(np.abs(X)), input_dropout))
    
    def init_classifier(w):
        optimizer_w = torch.optim.Adam(w.parameters())
        bce_loss_fn = torch.nn.BCEWithLogitsLoss()
        
        for j in range(1000):
            
            optimizer_w.zero_grad()
            idx = np.arange(0,X_torch.shape[0])
            np.random.shuffle(idx)
            X_batch, Y_batch = X_torch[idx[:batch_size]], Y_torch[idx[:batch_size]]
            
            loss_w = get_loss_fn(X_batch, Y_batch, w, symmetric(beta), optimize_beta = False)
            loss_w.backward()
            optimizer_w.step()
            if j % 100 == 0:
                print("Pretraining w...Loss is {}".format(loss_w))       
        return w
    
    def get_loss_fn(X,Y,alpha,beta, bce_loss_fn, optimize_beta=False):
        I = torch.eye(X.shape[1]).to(device)
        
        if not learn_proj:
            bce = bce_loss_fn(alpha(X@(I - beta)).squeeze(), Y)
        else:
            bce = bce_loss_fn(alpha(X@beta).squeeze(), Y)
        if optimize_beta:
            bce = -bce
            
        return bce

    def solve_linear_minimization(grad, iters=15):

         s = torch.randn_like(grad)
         s.requires_grad = True
         optimizer = torch.optim.Adam([s]) #torch.optim.SGD([s], lr = 1e-3, momentum = 0.8)
         for i in range(iters):
                optimizer.zero_grad()
                loss = (grad*s).mean()
                loss.backward()
                optimizer.step()
           
                # project
            
                D,U = torch.linalg.eigh(symmetric(s))
                D = D.detach().cpu().numpy()
                D_plus_diag = solve_constraint(D,d=d)
                D = torch.tensor(np.diag(D_plus_diag).real).float()#.to(device)
                s.data = U@D@U.T
         return s.to(device)
    
    # initalize beta with the INLP solution
    
    X_torch = torch.tensor(X).float().to(device)
    Y_torch = torch.tensor(Y).float().to(device)
    
    num_labels = len(set(Y.tolist()))
    if num_labels == 2:
        w = torch.nn.Linear(X.shape[1], 1).to(device) #torch.randn(X.shape[1]).float().to(device)*1e-1
    else:
        w = torch.nn.Linear(X.shape[1], num_labels).to(device)
        
        
    if init_beta is not None:
        Q = torch.randn(d, X.shape[1])*1e-3
        Q = Q / Q.norm(dim=1, keepdim=True)
        beta = torch.tensor(init_beta).float().to(device)
    else:
        #A = torch.tensor(w.weight.data.detach().cpu().numpy()).float().to(device)
        #A = A / A.norm(dim=1, keepdim=True)
        #beta = A.T@A
        beta = symmetric(torch.rand(X.shape[1], X.shape[1])).to(device)*1e-1        
        
    #w = init_classifier(w)

    w.requires_grad = True
    beta.requires_grad = True
    #optimizer_w = torch.optim.Adam(w.parameters())
    #optimizer_adv = torch.optim.Adam([beta], lr = lr)
    if use_adam:
        optimizer_w = torch.optim.Adam(w.parameters())
        optimizer_adv = torch.optim.Adam([beta], lr = lr)
    else:
        optimizer_w = torch.optim.SGD(w.parameters(), lr = lr, momentum = momentum)
        optimizer_adv = torch.optim.SGD([beta],  lr = lr, momentum = momentum, weight_decay=weight_decay) #, momentum = 0.0)

    if num_labels == 2:
        bce_loss_fn = torch.nn.BCEWithLogitsLoss()
        Y_torch = Y_torch.float()
    else:
        bce_loss_fn = torch.nn.CrossEntropyLoss()
        Y_torch = Y_torch.long()
        
    ws = [w.weight.data.detach().cpu().numpy().copy()]
    advs = [beta.detach().cpu().numpy().copy()]
    
    grads_beta = []
    grads_alpha = []
    best_adv, best_score = None, 1
    count_examples = 0.
    if replay:
        prev_clfs = []
    
    lr = 1e-2
    #maj = max(Y.mean(),1-Y.mean())
    from collections import Counter
    c = Counter(Y)
    fracts = [v/sum(c.values()) for v in c.values()]
    maj = max(fracts)
    if verbose:
        print("Majority accuracy is {}".format(maj))
    
    dropout = torch.nn.Dropout(input_dropout)
    with open("accs.txt".format(d), "w") as f:
    # pbar = tqdm.tqdm_notebook(range(out_iters))
    
     print("starting...")
     for i in range(out_iters):
                
        for j in range(in_iters_adv):
            #print(i,j)
            beta = symmetric(beta)
            optimizer_adv.zero_grad()
            
            idx = np.arange(0,X_torch.shape[0])
            np.random.shuffle(idx)
            X_batch, Y_batch = X_torch[idx[:batch_size]], Y_torch[idx[:batch_size]]
            with torch.no_grad():
                X_batch += torch.randn_like(X_batch).to(device)*noise_std
            X_batch = dropout(X_batch)
            
            loss_adv = get_loss_fn(X_batch, Y_batch, w if (not replay or not prev_clfs) else random.choice(prev_clfs), symmetric(beta), bce_loss_fn, optimize_beta = True)
            loss_adv.backward()
            optimizer_adv.step() 
            grads_beta.append(beta.grad.detach().cpu().norm().numpy())
            
            # project
            
            if i % project_freq == 0:
                  
                with torch.no_grad():
                    D,U = torch.linalg.eigh(symmetric(beta).detach().cpu())
                    D = D.detach().cpu().numpy()
                    D_plus_diag = solve_constraint(D,d=d)
                    D = torch.tensor(np.diag(D_plus_diag).real).float().to(device)
                    U = U.to(device)
                    beta.data = U@D@U.T 
                    advs.append(beta.detach().cpu().numpy().copy())
  
        
        for j in range(in_iters_clf):
            
            optimizer_w.zero_grad()
            idx = np.arange(0,X_torch.shape[0])
            np.random.shuffle(idx)
            X_batch, Y_batch = X_torch[idx[:batch_size]], Y_torch[idx[:batch_size]]
            
            loss_w = get_loss_fn(X_batch, Y_batch, w, symmetric(beta), bce_loss_fn, optimize_beta = False)
            loss_w.backward()
            optimizer_w.step()
            
            grads_alpha.append(w.weight.grad.detach().cpu().norm().numpy())


        if replay:
            prev_clfs.append(copy.deepcopy(w))
        if replay and len(prev_clfs) > 100:
            prev_clfs = prev_clfs[-100:]
            
        count_examples += batch_size
        
        if count_examples%evalaute_every ==0:
            ws.append(w.weight.data.detach().cpu().numpy().copy())

        #if count_examples % 500 == 0:
        #    advs.append(symmetric(beta).detach().cpu().numpy().copy())
        #print("==============================================")
        if i % evalaute_every == 0:
            
            if verbose:
                diffs_ws = np.mean([np.linalg.norm(ws[i]-ws[i-1]) for i in range(1,len(ws))])
                diffs_advs = np.mean([np.linalg.norm(advs[i]-advs[i-1]) for i in range(1,len(advs))])
            
            ws = ws[-1000:]
            advs = advs[-100:]
            
            if verbose:
                print("{}/{}".format(i, out_iters))
                #print(i, "delta w: {}; delta adv: {}".format(diffs_ws, diffs_advs))
                print("grad beta", np.mean(grads_beta))
                print("grad alpha", np.mean(grads_alpha))
            score = get_score(X,Y, advs[-1], d)
            
            #f.write("iteartion: {}; score: {}; grad_w: {}; grad_beta: {}\n".format(i, score, np.mean(grads_alpha),
            #                                                                      np.mean(grads_beta)))
            if np.abs(score-maj) < np.abs(best_score-maj):
                best_adv, best_score = symmetric(beta).detach().cpu().numpy().copy(), score
                
            # update progress bar
            #pbar.set_description("Acc post-projection: {:.3f} %; best so-far: {:.3f} %; Gap: {:.3f} %; Maj: {:.3f} %".format(  score*100, min(best_score, score)*100,  np.abs(best_score-maj)*100, maj*100  ))
            print("{}/{}. Acc post-projection: {:.3f} %; best so-far: {:.3f} %; Gap: {:.3f} %; Maj: {:.3f} %".format(i, out_iters,  score*100, min(best_score, score)*100,  np.abs(best_score-maj)*100, maj*100  ))
            #pbar.refresh() # to show immediately the update
            #time.sleep(0.01)
            
            grads_alpha = []
            grads_beta = []
            
            if i > 1 and np.abs(best_score - maj) < epsilon:
                advs.append(symmetric(beta).detach().cpu().numpy().copy())
                return ws, advs, best_adv, best_score
    if verbose or True:
        print("=============================")
        print("Best score is: {}".format(best_score))
    advs.append(symmetric(beta).detach().cpu().numpy().copy())
    return ws, advs, best_adv, best_score
    

if __name__ == "__main__":

    n,d = 5000,1024
    X = np.random.randn(n,d)
    Y = (np.random.rand(n) > 0.5).astype(int)
  
    X[:,0] = (Y+np.random.randn(*Y.shape)*0.3)**2 + 0.3*Y
    X[:,1] = (Y+np.random.randn(*Y.shape)*0.1)**2 - 0.7*Y
    X[:,2] = (Y+np.random.randn(*Y.shape)*0.3)**2 + 0.5*Y + np.random.randn(*Y.shape)*0.2
    X[:,3] = (Y+np.random.randn(*Y.shape)*0.5)**2 - 0.7*Y + np.random.randn(*Y.shape)*0.1
    
    ws, advs, best_adv = solve_fantope_relaxation(X,Y,d=1,init_beta=None,device="cpu", out_iters=50000, in_iters=1)
