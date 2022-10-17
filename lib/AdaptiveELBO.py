
########################################################
# Copyright Vanessa Gomez Verdejo: vanessag@ing.uc3m.es
########################################################
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0
########################################################
# This code is based on the pyro.infer.trace_elbo class
# of Uber Technologies and borrowed from: 
# https://docs.pyro.ai/en/stable/_modules/pyro/infer/trace_elbo.html
########################################################


'''These functions implement the adaptive ELBO implemention to 
   run the Adaptive VSGP based on VSI proposed in the Apenddix I of: 
   V. Gómez-Verdejo and M. Martínez-Ramón, "Adaptive Sparse Gaussian Process".
   Submitted to Pattern Recognition. 2022.'''
        
import sys

from pyro.distributions.util import scale_and_mask
from pyro.poutine.util import is_validation_enabled
from pyro.util import warn_if_inf, warn_if_nan
import pyro.poutine as poutine

def compute_log_prob_time(model_trace, forget_factor):
  result = 0.0
  for name, site in model_trace.nodes.items():
      if site["type"] == "sample":
          if "log_prob_sum" in site:
              log_p = site["log_prob_sum"]
          else:
              try:
                  if site['name'] == 'likelihood.y':
                    log_p = site["fn"].base_dist.log_prob(site["value"])
                    log_p = forget_factor @ log_p
                    #log_p  = torch.sum(log_p)
                  else:
                    log_p = site["fn"].log_prob(
                      site["value"], *site["args"], **site["kwargs"]
                  )
              except ValueError as e:
                  _, exc_value, traceback = sys.exc_info()
                  shapes = model_trace.format_shapes(last_site=site["name"])
                  raise ValueError(
                      "Error while computing log_prob_sum at site '{}':\n{}\n{}\n".format(
                          name, exc_value, shapes
                      )
                  ).with_traceback(traceback) from e
              log_p = scale_and_mask(log_p, site["scale"], site["mask"]).sum()
              site["log_prob_sum"] = log_p
              if is_validation_enabled():
                  warn_if_nan(log_p, "log_prob_sum at site '{}'".format(name))
                  warn_if_inf(
                      log_p,
                      "log_prob_sum at site '{}'".format(name),
                      allow_neginf=True,
                  )
          result = result + log_p
  
  return result


def adaptive_elbo(model, guide, forget_factor, *args, **kwargs):
    # run the guide and trace its execution
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
    # run the model and replay it against the samples from the guide
    model_trace = poutine.trace(
        poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
    # Compute log prob with forgetting factor    
    time_log_p = compute_log_prob_time(model_trace, forget_factor)
    
    # construct the elbo loss function
    return -1*(time_log_p - guide_trace.log_prob_sum())


# note that simple_elbo takes a model, a guide, and their respective arguments as inputs
def adaptive_elbo2(model, guide,forget_factor, *args, **kwargs):
    # run the guide and trace its execution
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
    # run the model and replay it against the samples from the guide
    model_trace = poutine.trace(
        poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
    # construct the elbo loss function
    return -1*(model_trace.log_prob_sum() - guide_trace.log_prob_sum())
