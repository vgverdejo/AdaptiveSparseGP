########################################################
# Copyright Vanessa Gomez Verdejo: vanessag@ing.uc3m.es
########################################################
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0
########################################################
# This code is based on the Sparse GP Regression model
# implementation of Uber Technologies and borrowed from: 
# https://docs.pyro.ai/en/stable/_modules/pyro/contrib/gp/models/gpr.html
########################################################

import numpy as np
import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.models.model import GPModel
from pyro.nn.module import PyroParam, pyro_method
from choldate import cholupdate, choldowndate       

#################################
# Some utilities for this class
#################################


def tensor_remove_row(A,p):
    return torch.cat((A[:p],A[p+1:]))

def tensor_remove_column(A,p):
    return torch.cat((A[:,:p],A[:,p+1:]),1)

def cholesky_add_row(L, a, aa):
    c = torch.linalg.solve_triangular( L, a, upper=False)
    d = torch.sqrt(aa-c.t()@c)
    L_new = torch.cat((L,c.t()))
    L_new = torch.cat((L_new,torch.zeros((L_new.shape[0],1))),dim=1)
    L_new[-1,-1] = d
    return L_new

def cholesky_remove_row(L, pos):
    L_new = torch.zeros_like(L[:-1,:-1])
    L_new[:pos,:pos] = L[:pos,:pos] 
    L_new[pos:,:pos] = L[pos+1:,:pos] 
    L33 = L[pos+1:,pos+1:] 
    l32 = L[pos+1:,pos] 
    L33_new = L33.detach().numpy().copy().astype(float)
    r = l32.detach().numpy().copy().astype(float)
    cholupdate(L33_new.T, r)
    L_new[pos:,pos:] = torch.from_numpy(L33_new)

    return L_new


########################################################
# The definition of the model
########################################################


class AdaptiveSparseGPRegression(GPModel):
    """
    Adaptive Sparse Gaussian Process Regression model.
    
    This class implements the algorithm proposed in
    Adaptive Sparse Gaussian Process Regression model proposed in:
        V. Gómez-Verdejo and M. Martínez-Ramón, "Adaptive Sparse Gaussian Process".
        Transactions on Neural Networks. 2023.

    :param torch.Tensor X: A input data for training. Its first dimension is the number
        of data points. First positions of X are the oldest and later positions the more recent.
    :param torch.Tensor y: An output data for training. Its last dimension is the
        number of data points. First positions of y are the oldest and later positions the more recent.
    :param ~pyro.contrib.gp.kernels.kernel.Kernel kernel: A Pyro kernel object, which
        is the covariance function :math:`k`.
    :param torch.Tensor Xu: Initial values for inducing points.
    :param float lamb: Forgetting factor (0<lamb<=1) used to include adaptive capabilities
        to the model.
    :param str model_mode: Parameter ('batch' or 'online') indicating the variational 
        inference used for the parameter learning.
    :param torch.Tensor noise: Variance of Gaussian noise of this model.
    :param callable mean_function: An optional mean function :math:`m` of this Gaussian
        process. By default, we use zero mean.
    
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    :param str name: Name of this model.
    """

    def __init__(
        self, X, y, kernel, Xu, lamb = 1, r = None, model_mode = 'batch', noise=None, mean_function=None, jitter=1e-6
    ):

        assert isinstance(
            X, torch.Tensor
        ), "X needs to be a torch Tensor instead of a {}".format(type(X))
        if y is not None:
            assert isinstance(
                y, torch.Tensor
            ), "y needs to be a torch Tensor instead of a {}".format(type(y))
        assert isinstance(
            Xu, torch.Tensor
        ), "Xu needs to be a torch Tensor instead of a {}".format(type(Xu))
        
        assert ((lamb>0) and (lamb<=1)), "lamb has to be greater than 0 and lower or equal to 1 instead of {}".format(lamb)
        
        assert ((model_mode=='batch') or (model_mode=='online')), "model_mode has to be 'batch' or 'online' instead of {}".format(model_mode)
        
        super().__init__(X, y, kernel, mean_function, jitter)  # Trick to pass the data control
        
        
        self.Xu = Parameter(Xu)
    
        if model_mode=='batch':
            # If we use batch learning, we can select learning this or not
            self.Xu = Parameter(Xu, requires_grad=False)  
    
        self.Xu_new = None

        noise = self.X.new_tensor(1.0) if noise is None else noise
        self.noise = PyroParam(noise, constraints.positive)

        # Include forgetting factor, we consider first positions of X, y are the oldest
        forget_factor =  torch.flip(torch.cumprod(lamb*torch.ones(X.shape[0]), dim=0), [0])
        self.Lambda = torch.cat((forget_factor[1:],torch.tensor([1.0])))
        self.lamb = lamb
        
        self.model_mode = model_mode
        

        # Flag to indicate if L has to be precomputed or not. This matrix is only used in .forward(), 
        # so we do not need to update it during the inference
        self.L_updated = False 
        # Precompute some parameters 
        self.N = self.X.size(0)
        self.M = self.Xu.size(0)
        #self.update_all_variables()
        self.update_variables_inference()
        
    def set_online_learning(self):
        self.model_mode='online'
        # In online mode Xu is not updated, we will only update Xu_new
        for n,k in self.named_parameters():
            if n == 'Xu':
                k.requires_grad = False
        
        
    def set_batch_learning(self, fixed_Xu = True):
        self.model_mode='batch'
        # In batch mode Xu is updated
        
        for n,k in self.named_parameters():
            if n == 'Xu':
                k.requires_grad = not fixed_Xu


    def model(self):
        # This function implements the generative proccess. 
        # The following equations are considered:
        # W = (inv(Luu) @ Kuf).T
        # Qff = Kfu @ inv(Kuu) @ Kuf = W @ W.T
        # trace_term = Lambda @ tr(Kff-Qff) / noise
        # noise_term = (sum(Lambda)-N) * log(2*pi*noise)
        # y_cov = W @ W.T + noise * inv(Lambda)
        # trace_term and noise terms are added into log_prob
        
        # We include to implementations:
        # 1) batch: uselful when the kernel parameters are updated, 
        #    since kernel matrix have to be recomputed and online 
        #    learning is not useful
        # 2) online: previous equations are implemented with one-rank
        #    updates stating from their previous values
        
        self.set_mode("model")


        if self.model_mode == 'batch':
            self.update_variables_inference()
            W = self.W
            D_lamb = self.D
            trace_term = self.trace_term 

        elif self.model_mode == 'online':
            if self.Xu_new is None:
                Xu_all = self.Xu
                M = self.Xu.size(0)
            else:
                Xu_all =  torch.cat((self.Xu, self.Xu_new))
                M = self.Xu.size(0) + 1

            # Compute new Kuu, Luu and Kuf parameters

            Kuu = self.kernel(Xu_all).contiguous()
            Kuu = 0.5*(Kuu+Kuu.T) # To ensure symmetry
            Kuu.view(-1)[:: M + 1] += self.jitter  # add jitter to the diagonal
            Luu = torch.linalg.cholesky(Kuu)

            Kuf = self.kernel(Xu_all, self.X)

            W = torch.linalg.solve_triangular(Luu, Kuf, upper=False).T

            D_lamb = self.noise/self.Lambda

            Kffdiag = self.kernel(self.X, diag=True)
            Qffdiag = W.pow(2).sum(dim=-1)

            trace_term = self.Lambda @ (Kffdiag - Qffdiag) / self.noise
            trace_term = trace_term.clamp(min=0)
        
        # Computing the likelihood is common for all the model modes
                
        # Add noise lambda dependent term
        noise_term = (torch.sum(self.Lambda)-self.Lambda.shape[0]) * torch.log(2*torch.pi* self.noise)
                
        zero_loc = self.X.new_zeros(self.N)
        f_loc = zero_loc + self.mean_function(self.X)
        
        if self.y is None:
            f_var = D_lamb + W.pow(2).sum(dim=-1)
            return f_loc, f_var
        else:
            pyro.factor(self._pyro_get_fullname("trace_term"), -(trace_term+noise_term)/ 2.0)
            
            return pyro.sample(
                self._pyro_get_fullname("y"),
                dist.LowRankMultivariateNormal(f_loc, W, D_lamb)
                .expand_by(self.y.shape[:-1])
                .to_event(self.y.dim() - 1),
                obs=self.y,
            ) 
    

    # Here, we define a set of auxiliar functions to update some variables
    def update_variables_inference(self):
        self.update_Luu()
        self.update_W_D()


    def update_Luu(self):
        '''When Xu is updated, we need to update Luu (Cholesky descomposition of Kuu)'''
        Kuu = self.kernel(self.Xu).contiguous()
        Kuu = 0.5*(Kuu+Kuu.T) # To ensure symmetry
        Kuu.view(-1)[:: self.M + 1] += self.jitter  # add jitter to the diagonal
        Luu = torch.linalg.cholesky(Kuu)  ### The computational cost of this is O(M^3)
        self.Luu = Luu
        
    def update_W_D(self):
        '''When either Xu or X is updated, we need to update Kuf, W, D and trace'''
        Kuf = self.kernel(self.Xu, self.X)
        W = torch.linalg.solve_triangular(self.Luu, Kuf, upper=False).t()
    
        self.Kuf = Kuf
        self.W = W 
        
        self.D = self.noise/self.Lambda
    
        Kffdiag = self.kernel(self.X, diag=True)
        self.Kffdiag = Kffdiag
        Qffdiag = self.W.pow(2).sum(dim=-1)
        self.dif_KQ = Kffdiag - Qffdiag

        trace_term = self.Lambda@self.dif_KQ / self.noise
        self.trace_term = trace_term.clamp(min=0)
        
    def update_L(self):
        '''When either Xu or X is updated, we need to update L'''
        W = self.W.t().contiguous()
        D = self.D
        W_Dinv = W / D
        K = W_Dinv.matmul(W.T).contiguous()
        K = 0.5*(K +K.T) # To ensure K is symmetric

        K.view(-1)[:: self.M + 1] += 1  # add identity matrix to K
        L = torch.linalg.cholesky(K)  #Computational burden of O((M)^3)
        self.L = L
        
    def get_L(self):
        if not self.L_updated:
            self.update_L()
            self.L_updated = True
        return self.L
    
    # Functions to online learning, to add/remove data and add/remove inducing points
    def add_new_data(self, X_new, y_new):
        '''We include a new training sample to the model'''

        self.X = torch.cat((self.X, X_new))
        self.y  =torch.cat((self.y, y_new))
       
        self.N = self.X.size(0)

        # Include new Lambda factor and multiply the previous with lamb
        self.Lambda = torch.cat((self.Lambda*self.lamb, torch.tensor([1.0])))
        
        # When X_new arrrives, we need to update Kuf, W, D, trace and L
        
        Kuf_new = self.kernel(self.Xu, X_new)
        W_new = torch.linalg.solve_triangular(self.Luu, Kuf_new, upper=False).t()
        self.Kuf = torch.cat((self.Kuf, Kuf_new),1) 
        self.W = torch.cat((self.W, W_new)) 
                
        self.D  = self.noise/self.Lambda
        
        Kffdiag_new = self.kernel(X_new, diag=True)
        Qffdiag_new = W_new.pow(2).sum(dim=-1)
        dif_KQ_new = Kffdiag_new - Qffdiag_new
        self.dif_KQ = torch.cat((self.dif_KQ, dif_KQ_new)) 
            
        trace_term = self.Lambda @ self.dif_KQ / self.noise
        self.trace_term = trace_term.clamp(min=0)
        
        self.Kffdiag = torch.cat((self.Kffdiag, Kffdiag_new))

        self.L_updated = False 

    def remove_data(self, L = None):
        '''We remove the last training samples from the model and keep L. 
        If L is not provided, we only remove the last data.
        We also update parameter depending on N, to limit them to L size
        Remaining parameters are not worthy to be updated (they are forgotten with lambda)'''
       
        if L is None:
            Ninit =1
        else:
            Ninit = self.N-L
        
        self.X = self.X[Ninit:]
        self.y = self.y[Ninit:]
        self.N = self.X.size(0)

        # Remove last Lambda factors
        self.Lambda =  self.Lambda[Ninit:]
        self.D = self.noise/self.Lambda
        
        # When we remove one data, we need to update Kuf, W, D, trace and L
        self.Kuf = self.Kuf[:,Ninit:] 
        #W_old = self.W[0]
        self.W = self.W[Ninit:]

        self.dif_KQ = self.dif_KQ[Ninit:] 
        trace_term = self.Lambda@self.dif_KQ / self.noise
        self.trace_term = trace_term.clamp(min=0)
       
        self.Kffdiag = self.Kffdiag[Ninit:]
        # self.LT_updated = False #We don't need to update LT because data are forgotten
        
    def add_new_Xu(self):
        '''We include the new training sample into the relevance vectors
          We implement online learning, so Xu is added with 
          one rank updates'''

        
        # We need to include Xu_new to the model   
        Xu_new = self.Xu_new.clone()
        Xu_aux = torch.cat((self.Xu.detach(), Xu_new))
        self.Xu = Parameter(Xu_aux, requires_grad=False)  
        self.M += 1   

        # Online update of Luu
        a = self.kernel(self.Xu[:-1], Xu_new)
        aa =  self.kernel(Xu_new) + self.jitter
        self.Luu = cholesky_add_row(self.Luu, a, aa)
        
        # Online update of Kuf, W, trace and D
        Kuf_new = self.kernel(Xu_new, self.X)
        self.Kuf = torch.cat((self.Kuf, Kuf_new))
       
        ####Luu_new @ W + d * W_new = Kuf_new  
        W_new = (Kuf_new - self.Luu[-1:,:-1]@ self.W.t())/ self.Luu[-1,-1]
        self.W = torch.cat((self.W, W_new.t()),1) 

        self.D  = self.noise/self.Lambda

        Qffdiag_new = self.W[:,-1:].pow(2).sum(dim=-1)
        self.dif_KQ -= Qffdiag_new
        trace_term = self.Lambda@self.dif_KQ / self.noise
        self.trace_term = trace_term.clamp(min=0)

        W_Dinv = self.W.t()/ torch.sqrt(self.D)  
        # W_new = W_Dinv[-1:]  and W_old = W_Dinv[:-1]     
        # K_new = [ K W_new@self.W, (W_new@W_old).T (W_new@W_new.T+1)]
        # Adding identity in last term of the diagonal
        # So, we update L adding one row
        #self.L = cholesky_add_row(self.L, W_Dinv[:-1]@W_Dinv[-1:].t(), W_Dinv[-1:]@W_Dinv[-1:].t()+1)
        self.L_updated = False 
        
        # Set Xu_new to None to avoid to be used later 
        self.Xu_new = None

 
    def remove_Xu(self, position = None):
        '''We remove relevance vectors
          We implement online learning, so Xu is removed with 
          one rank updates'''
        
        # Remove from Xu
        Xu_aux = self.Xu.detach()
        ind_all = np.arange(Xu_aux.shape[0])
        ind_keep = np.setdiff1d(ind_all, position) 
        self.Xu = Parameter(Xu_aux[ind_keep], requires_grad=False)
        self.M = len(ind_keep)
        
        for pos_Xu in position:
            # Online update of Luu
            self.Luu = cholesky_remove_row(self.Luu, pos_Xu)
            
            # Online update of Kuf, W, trace and D
            self.Kuf = tensor_remove_row(self.Kuf, pos_Xu)
           
        # Update W
        self.W = torch.linalg.solve_triangular(self.Luu, self.Kuf, upper=False).t()
    
        Qffdiag = self.W.pow(2).sum(dim=-1)
        self.dif_KQ = self.Kffdiag - Qffdiag
            
        trace_term = self.Lambda@self.dif_KQ / self.noise
        self.trace_term = trace_term.clamp(min=0)
    

        # We could try to do this by updating only the rows of L down to the removed Xu,
        # on average, we can apply M/2 rank one updates.
        # But for the sake of simplicity, we think native torch.linalg.cholesky() implementation
        # it's going to be faster and it is computed when needed   
        self.L_updated = False 

    def update_noise(self):
        '''When the noise parameter is updated, we need to update D and L'''
        self.D  = self.noise/self.Lambda
        self.L_updated = False 

    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self._load_pyro_samples()
    
    
    def forward(self, Xnew, full_cov=False, noiseless=True):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, X_u, \epsilon) = \mathcal{N}(loc, cov).

        .. note:: The noise parameter ``noise`` (:math:`\epsilon`), the inducing-point
            parameter ``Xu``, together with kernel's parameters have been learned from
            a training procedure (MCMC or SVI).

        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :param bool noiseless: A flag to decide if we want to include noise in the
            prediction output or not.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")

        # W = inv(Luu) @ Kuf
        # Ws = inv(Luu) @ Kus
        # D as in self.model()
        # K = I + W @ inv(D) @ W.T = L @ L.T
        # S = inv[Kuu + Kuf @ inv(D) @ Kfu]
        #   = inv(Luu).T @ inv[I + inv(Luu)@ Kuf @ inv(D)@ Kfu @ inv(Luu).T] @ inv(Luu)
        #   = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
        #   = inv(Luu).T @ inv(K) @ inv(Luu)
        #   = inv(Luu).T @ inv(L).T @ inv(L) @ inv(Luu)
        # loc = Ksu @ S @ Kuf @ inv(D) @ y = Ws.T @ inv(L).T @ inv(L) @ W @ inv(D) @ y
        # cov = Kss - Ksu @ inv(Kuu) @ Kus + Ksu @ S @ Kus
        #     = kss - Ksu @ inv(Kuu) @ Kus + Ws.T @ inv(L).T @ inv(L) @ Ws

       
        N = self.X.size(0)
        M = self.Xu.size(0)
                
        W =  self.W.t().contiguous()
        D = self.D
        
        W_Dinv = W / D

        # get y_residual and convert it into 2D tensor for packing
        y_residual = self.y - self.mean_function(self.X)
        y_2D = y_residual.reshape(-1, N).t()
        
        W_Dinv_y = W_Dinv.matmul(y_2D)
        

        # End caching ----------

        Kus = self.kernel(self.Xu, Xnew)
        Ws = torch.linalg.solve_triangular(self.Luu, Kus, upper=False)
        
        pack = torch.cat((W_Dinv_y, Ws), dim=1)
        L = self.get_L()
        Linv_pack = torch.linalg.solve_triangular(L, pack, upper=False)
        # unpack
        Linv_W_Dinv_y = Linv_pack[:, : W_Dinv_y.shape[1]]
        Linv_Ws = Linv_pack[:, W_Dinv_y.shape[1] :]

        C = Xnew.size(0)
        loc_shape = self.y.shape[:-1] + (C,)
        loc = Linv_W_Dinv_y.t().matmul(Linv_Ws).reshape(loc_shape)
        if full_cov:
            Kss = self.kernel(Xnew).contiguous()
            if not noiseless:
                Kss.view(-1)[:: C + 1] += self.noise  # add noise to the diagonal
            Qss = Ws.t().matmul(Ws)
            cov = Kss - Qss + Linv_Ws.t().matmul(Linv_Ws)
            cov_shape = self.y.shape[:-1] + (C, C)
            cov = cov.expand(cov_shape)
        else:
            Kssdiag = self.kernel(Xnew, diag=True)
            if not noiseless:
                Kssdiag = Kssdiag + self.noise
            Qssdiag = Ws.pow(2).sum(dim=0)
            cov = Kssdiag - Qssdiag + Linv_Ws.pow(2).sum(dim=0)
            cov_shape = self.y.shape[:-1] + (C,)
            cov = cov.expand(cov_shape)

        return loc + self.mean_function(Xnew), cov
    
    
    def batch_update(self, fixed_Xu = True, num_steps = 100, lr =0.05, verbose = False):
  
        ''' Function to batch learning of the model with data stored in self.X and self.y'''

        # Set batch mode according fixed_Xu
        self.set_batch_learning(fixed_Xu = fixed_Xu)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

        loss = None
        for i in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(self.model, self.guide)
            loss.backward() 
            optimizer.step()

        # Besides, we have updated the model variables
        self.update_variables_inference()

        if verbose:
            print('We infer the noise parameter. New noise value: %2.4f' % self.noise.data.numpy())
            print('We infer the kernel parameters. New kernel parameters:')
            print('Kernel lengthscale: %2.2f'%self.kernel.lengthscale.item())
            print('Kernel amplitude: %2.2f'%self.kernel.variance.item())   
            print('_____________')

        return loss
    
    
    
    def online_update(self, X_new, y_new, L = 100, M = 10, perc_th = 0.01, num_steps = 20, lr =0.05, verbose = False):
  
        ''' Function to online update model for each new data'''

        assert isinstance(
            X_new, torch.Tensor
            ), "X_new needs to be a torch Tensor instead of a {}".format(type(X_new))

        assert (X_new.shape[0] == 1) , "The number of data in X_new (number of rows) has to be 1 instead of a {}".format(X_new.shape[0])

        assert isinstance(
            y_new, torch.Tensor
            ), "y_new needs to be a torch Tensor instead of a {}".format(type(y_new))

        assert (y_new.shape[0] == 1) , "The number of data in y_new (number of rows) has to be 1 instead of a {}".format(y_new.shape[0])

        # Check online mode, if not, set it!
        if self.model_mode != 'online':
            self.set_online_learning()

        # Each iteration, add and remove a new data according to T value
        if self.y.shape[0]>(L-1):
            self.remove_data(L-1) 

        self.add_new_data(X_new, y_new)

        # Analyze the relevance vector importances and check to remove inducing points
        Qff_relevance = self.Lambda @ self.W.pow(2) # Rm (relevance of each inducing point)
        th = perc_th * torch.max(Qff_relevance) # The theshold is set as a percentage of the maximum relevance 
        pos_remove, _  = torch.where(Qff_relevance < th)[0].sort(descending = True)
        if (pos_remove.shape[0]== 0) and (self.M >= M):
            pos_remove = torch.tensor([torch.argmin(Qff_relevance)])

        # Less relevant inducing points are removed
        #model.remove_Xu(position = pos_remove)
        # We only remove them from Xu because during the inference kernel parameters are updated,
        # so Kuu, Kuf, W.... has to be recomputed 
        Xu_aux = self.Xu.detach()
        ind_all = np.arange(Xu_aux.shape[0])
        ind_keep = np.setdiff1d(ind_all, pos_remove) 
        self.Xu = Parameter(Xu_aux[ind_keep], requires_grad=False)
        self.M = len(ind_keep)

        if verbose and (pos_remove.shape[0]>0):
            print('We have removed %d inducing points. The new set of inducing points are: '%pos_remove.shape[0])
            print(self.Xu.data.numpy())


        # Initialize Xu for the inference
        self.Xu_new = Parameter(X_new.clone())

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

        loss = None
        for i in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(self.model, self.guide)
            loss.backward()  
            optimizer.step()


        # We need to include Xu_new to the model   
        Xu_aux = torch.cat((self.Xu.detach(), self.Xu_new.clone()))
        self.Xu = Parameter(Xu_aux, requires_grad=False)  
        self.M += 1   


        # Besides, we have updated the kernel or noise parameters, so we have to update the model 
        self.update_variables_inference()


        if verbose:
            print('We infer the noise parameter. New noise value: %2.4f' % self.noise.data.numpy())
            print('We infer the kernel parameters. New kernel parameters:')
            print('Kernel lengthscale: %2.2f'%self.kernel.lengthscale.item())
            print('Kernel amplitude: %2.2f'%self.kernel.variance.item())   
            print('_____________')

        return loss
    
    def fast_online_update(self, X_new, y_new, L = 100, M = 10, perc_th = 0.01,  trace_th = None, verbose = False):

        ''' Function to fast online update model for each new data'''

        assert isinstance(
            X_new, torch.Tensor
            ), "X_new needs to be a torch Tensor instead of a {}".format(type(X_new))

        assert (X_new.shape[0] == 1) , "The number of data in X_new (number of rows) has to be 1 instead of a {}".format(X_new.shape[0])

        assert isinstance(
            y_new, torch.Tensor
            ), "y_new needs to be a torch Tensor instead of a {}".format(type(y_new))

        assert (y_new.shape[0] == 1) , "The number of data in y_new (number of rows) has to be 1 instead of a {}".format(y_new.shape[0])

 
        # Check online mode, if not, set it!
        if self.model_mode != 'online':
            self.set_online_learning()

        # Each iteration, add and remove a new data according to T value
        if self.y.shape[0]>(L-1):
            self.remove_data(L-1) 

        self.add_new_data(X_new, y_new)


        # Check if a new Xu needs to be added
        if trace_th is None:
            trace_th = self.Kffdiag.sum()/L
  
        # When the trace increases, we add an inducing point
        if self.trace_term.item()>trace_th:         
            # Add Xu without inference
            self.Xu_new = X_new.clone()
            self.add_new_Xu()
    
        # Analyze the relevance vector importances and check to remove inducing points 
        Qff_relevance = self.Lambda @ self.W.pow(2) # Rm (relevance of each inducing point)
        th = perc_th * torch.max(Qff_relevance) # The theshold is set as a percentage of the maximum relevance 
        pos_remove, _  = torch.where(Qff_relevance < th)[0].sort(descending = True)
        if (pos_remove.shape[0]== 0) and (self.M >= M):
            pos_remove = torch.tensor([torch.argmin(Qff_relevance)])

        if (pos_remove.shape[0]> 0):
            # Less relevant inducing points are removed
            self.remove_Xu(position = pos_remove)

        if verbose and (pos_remove.shape[0]>0):
            print('We have removed %d inducing points. The new set of inducing points are: '%pos_remove.shape[0])
            print(self.Xu.data.numpy())



