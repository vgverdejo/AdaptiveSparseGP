########################################################
# Copyright Vanessa Gomez Verdejo: vanessag@ing.uc3m.es
########################################################
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

'''These functions implement some utilities to run the Fast-AGP and
   AGP models proposed in: 
   V. Gómez-Verdejo and M. Martínez-Ramón, "Adaptive Sparse Gaussian Process".
   Submitted to Pattern Recognition. 2022.'''


import torch
import pyro
from torch.nn import Parameter
from AdaptiveSparseGPRegression import tensor_remove_row

def model_update_AGP(model, X_new, y_new, T = 100, M = 10, perc_th = 0.01, optimizer = None, num_steps = 20, verbose = False):
  
  ''' Function to update the AGP model for each new data'''

  assert isinstance(
      X_new, torch.Tensor
  ), "X_new needs to be a torch Tensor instead of a {}".format(type(X_new))

  assert (X_new.shape[0] == 1) , "The number of data in X_new (number of rows) has to be 1 instead of a {}".format(X_new.shape[0])
  
  assert isinstance(
      y_new, torch.Tensor
      ), "y_new needs to be a torch Tensor instead of a {}".format(type(y_new))

  assert (y_new.shape[0] == 1) , "The number of data in y_new (number of rows) has to be 1 instead of a {}".format(y_new.shape[0])

  loss = None 
  if optimizer is None:
    torch.optim.Adam(model.parameters(), lr=0.05)

  loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    
  # Each iteration, add and remove a new data according to T value
  if model.y.shape[0]>(T-1):
    model.remove_data()
  model.add_new_data(X_new, y_new, update_L = False)

  # Analyze the relevance vector importances and check to remove inducing points
  Qff_relevance = model.Lambda@model.W.pow(2) # Rm (relevance of each inducing point)
  th = perc_th * torch.max(Qff_relevance) # The theshold is set as a percentage of the maximum relevance 
  pos_remove, _  = torch.where(Qff_relevance < th)[0].sort(descending = True)
  if (pos_remove.shape[0]== 0) and (model.M ==M):
    pos_remove = torch.tensor([torch.argmin(Qff_relevance)])

  # Less relevant inducing points are removed
  #model.remove_Xu(position = pos_remove)
  for pos_Xu in pos_remove:
    model.Xu = tensor_remove_row(model.Xu,pos_Xu)
    model.M -= 1   

  if verbose and (pos_remove.shape[0]>0):
    print('We have removed %d inducing points. The new set of inducing points are: '%pos_remove.shape[0])
    print(model.Xu.data.numpy())

         
  # Initialize Xu for the inference
  model.Xu_new = Parameter(X_new.clone())
  
  model.model_mode = 'online'
  for i in range(num_steps):
      optimizer.zero_grad()
      loss = loss_fn(model.model, model.guide)
      loss.backward(retain_graph=True)
      optimizer.step()
    
  
  # We need to include Xu_new to the model   
  #model.add_new_Xu()
  model.Xu = torch.cat((model.Xu,model.Xu_new.clone()))
  model.M += 1   

  # Besides, we have updated the kernel or noise parameters we have to update the model 
  model.update_all_variables()
 
  
  if verbose:
    if Noise_infer:  
      print('We infer the noise parameter. New noise value: %2.4f' % model.noise.data.numpy())
    if Kernel_infer:
      print('We infer the kernel parameters. New kernel parameters:')
      print('Kernel lengthscale: %2.2f'%model.kernel.lengthscale.item())
      print('Kernel amplitude: %2.2f'%model.kernel.variance.item())
    
    print('_____________')
  
  return loss


def model_update_FastAGP(model, X_new, y_new, T = 100, M = 10, perc_th = 0.01,  trace_th = None, verbose = False):

  ''' Function to update the Fast-AGP model for each new data'''

  assert isinstance(
      X_new, torch.Tensor
  ), "X_new needs to be a torch Tensor instead of a {}".format(type(X_new))

  assert (X_new.shape[0] == 1) , "The number of data in X_new (number of rows) has to be 1 instead of a {}".format(X_new.shape[0])
  
  assert isinstance(
      y_new, torch.Tensor
      ), "y_new needs to be a torch Tensor instead of a {}".format(type(y_new))

  assert (y_new.shape[0] == 1) , "The number of data in y_new (number of rows) has to be 1 instead of a {}".format(y_new.shape[0])

  # Each iteration, add and remove a new data according to T value and 
  # update model parameters with new data 
  if model.y.shape[0]>(T-1):
    model.remove_data()
  model.add_new_data(X_new, y_new)

  # Check if a new Xu needs to be added
  if trace_th is None:
    trace_th = model.Kffdiag.sum()/T
  
  # When the trace increases, we add an inducing point
  if model.trace_term.item()>trace_th:         
      # Add Xu without inference
      model.Xu_new = X_new.clone()
      model.add_new_Xu()


  # Analyze the relevance vector importances and check to remove inducing points
  Qff_relevance = model.Lambda@model.W.pow(2) # Rm (relevance of each inducing point)
  th = perc_th * torch.max(Qff_relevance) # The theshold is set as a percentage of the maximum relevance 
  pos_remove, _  = torch.where(Qff_relevance < th)[0].sort(descending = True)
  if (pos_remove.shape[0]== 0) and (model.M >M):
    pos_remove = torch.tensor([torch.argmin(Qff_relevance)])

  # Less relevant inducing points are removed
  model.remove_Xu(position = pos_remove)

  if verbose and (pos_remove.shape[0]>0):
    print('We have removed %d inducing points. The new set of inducing points are: '%pos_remove.shape[0])
    print(model.Xu.data.numpy())
    



