import os,sys
import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm
import torch
import torch.distributions as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

import math
from torch.nn import init
from functools import partial
from torch.nn.modules.utils import _pair

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch import distributions
from torch.distributions.studentT import StudentT

import math
import numpy as np
from torch.nn import init
from functools import partial
from math import gamma as gamma_function

########################################################################################################################

def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return

########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

########################################################################################################################

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

########################################################################################################################

def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean=0
    std=0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for image, _ in loader:
        mean+=image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded=mean.view(mean.size(0),mean.size(1),1,1).expand_as(image)
    for image, _ in loader:
        std+=(image-mean_expanded).pow(2).sum(3).sum(2)

    std=(std/(len(dataset)*image.size(2)*image.size(3)-1)).sqrt()

    return mean, std

########################################################################################################################

def fisher_matrix_diag(t,x,y,model,criterion,sbatch=5, pass_t = False):
    # Init
    fisher={}
    if(x.size(0) < 500):
        sbatch = 1
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()
    samples_taken = 0
    for i in tqdm(range(0,x.size(0),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+1,x.size(0)]))).cuda()
        images=torch.autograd.Variable(x[b],volatile=False)
        target=torch.autograd.Variable(y[b],volatile=False)
        # Forward and backward
        model.zero_grad()
        if not pass_t:
            outputs=model.forward(images)
        else:
            outputs=model.forward(t, images)
        loss=criterion(t,outputs[t],target)
        loss.backward()
        samples_taken += 1
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=1*p.grad.data.pow(2)
    # Mean
    for n,_ in model.named_parameters():
        fisher[n]=fisher[n]/samples_taken
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher

def l2_reg(appr):
    loss_reg=0
    for name,param in appr.model.named_parameters():
        loss_reg+=torch.sum(param**2)/2

    return loss_reg/80

########################################################################################################################

def cross_entropy(outputs,targets,exp=1,size_average=True,eps=1e-5):
    out=torch.nn.functional.softmax(outputs)
    tar=torch.nn.functional.softmax(targets)
    if exp!=1:
        out=out.pow(exp)
        out=out/out.sum(1).view(-1,1).expand_as(out)
        tar=tar.pow(exp)
        tar=tar/tar.sum(1).view(-1,1).expand_as(tar)
    out=out+eps/out.size(1)
    out=out/out.sum(1).view(-1,1).expand_as(out)
    ce=-(tar*out.log()).sum(1)
    if size_average:
        ce=ce.mean()
    return ce

########################################################################################################################

def set_req_grad(layer,req_grad):
    if hasattr(layer,'weight'):
        layer.weight.requires_grad=req_grad
    if hasattr(layer,'bias'):
        layer.bias.requires_grad=req_grad
    return

########################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
########################################################################################################################


def compute_kl_g(mean, exp_var, prior_mean, prior_exp_var, q, v, sum = True, lamb = 1, initial_prior_var = 1):
    #print("mean shape:", mean.shape)
    #print("exp_var shape:", exp_var.shape)
    #print("prior_mean shape:", prior_mean.shape)
    #print("prior_exp_var shape:", prior_exp_var.shape)
    trace_term = torch.exp(exp_var - prior_exp_var)
    if lamb != 1:
        mean_term =  (mean - prior_mean)**2 * (lamb * torch.clamp(torch.exp(-prior_exp_var) - (1/initial_prior_var), min = 0.0) + (1/initial_prior_var))
    else:
        mean_term =  (mean - prior_mean)**2 * torch.exp(-prior_exp_var)
    det_term = prior_exp_var - exp_var
    
    if sum:
        return 0.5 * torch.sum(trace_term + mean_term + det_term - 1)
    else:
        return 0.5 * (trace_term + mean_term + det_term - 1)


#extend compute kl method to choose renyi between gaussians, renyi between q-gaussians, kl between q gaussians or kl begtween gaussians

def compute_re_g(mean, log_var, prior_mean, log_prior_var, alpha, v, sum = True, lamb = 1, initial_prior_var = 1):
    """
    Compute the Rényi divergence between univariate Gaussian posterior and prior distributions.
    
    Args:
        mean (torch.Tensor): Posterior mean (shape: [n]).
        log_var (torch.Tensor): Log-variance of the posterior (shape: [n]).
        prior_mean (torch.Tensor): Prior mean (shape: [n]).
        log_prior_var (torch.Tensor): Log-variance of the prior (shape: [n]).
        alpha (float): Order of the Rényi divergence (alpha > 0, alpha != 1).
    
    Returns:
        torch.Tensor: Rényi divergence for each element in the input tensor (shape: [n]).
    """
    # Compute posterior variance and prior variance from log-variances
    var = torch.exp(log_var)  # Posterior variance
    prior_var = torch.exp(log_prior_var)  # Prior variance 
    # to check
    prior_var_lamda = (lamb * torch.clamp(torch.exp(-log_prior_var) - (1/initial_prior_var), min = 0.0) + (1/initial_prior_var))

    # Compute the mixed variance (Σ_α)^*
    mixed_var = alpha * prior_var + (1 - alpha) * var

    # Compute the Mahalanobis term: α (μ - μ_prior)^2 / (Σ_α)^*
    if lamb != 1:
        mahalanobis_term = alpha * (mean - prior_mean) ** 2 /(alpha * prior_var_lamda + (1 - alpha) * var)
    
    else:
        mahalanobis_term = alpha * (mean - prior_mean) ** 2 / mixed_var

    # Compute the determinant term: log(|Σ_α^*|) - ((1 - α) log(|Σ|) + α log(|Σ_prior|))
    log_det_term = torch.log(mixed_var) - ((1 - alpha) * log_var + alpha * log_prior_var)

    if sum:
        return 0.5 * torch.sum(mahalanobis_term - 0.5 / (alpha - 1) * log_det_term)
    # Combine terms to compute Rényi divergence
    else:
        return 0.5 * mahalanobis_term - 0.5 / (alpha - 1)

def compute_psi_vectorized(v, log_sigma):
    """
    Compute Ψ for the d-dimensional case using degrees of freedom v.
    Works with log-space sigma.
    
    Parameters:
    -----------
    v : float
        Degrees of freedom (v > 0)
    log_sigma : torch.Tensor
        Log variance parameter of shape (d,)
        
    Returns:
    --------
    torch.Tensor
        The computed Ψ values of shape (d,)
    """
    k = log_sigma.shape[0]

    # Ensure k does not equal v
    if k == v:
        raise ValueError("k cannot equal v")
    # Compute numerator: Γ((v+1)/2)
    numerator = gamma_function((v + 1) / 2)
    
    # Compute denominator parts
    pi_v_term = (math.pi * v)**(k/2)
    gamma_term = gamma_function(v/2)
    
    # Convert to tensor for broadcasting
    numerator = torch.tensor(numerator, dtype=torch.float32)
    pi_v_term = torch.tensor(pi_v_term, dtype=torch.float32)
    gamma_term = torch.tensor(gamma_term, dtype=torch.float32)
    
    # Use log_sigma directly and exp(log_sigma/2) for sqrt(sigma)
    sigma_term = torch.prod(torch.exp(log_sigma/2)) #computes the determinant od a diagonal matrix
    
    # Combine terms
    base = numerator / (pi_v_term * gamma_term * sigma_term)
    
    # Raise to power -2/(v+1)
    psi = base ** (-2/(v + k))
    
    return psi

def compute_psi_vectorized_ln(v, k):
    """
    Compute Ψ for the d-dimensional case using log gamma functions.
    
    Parameters:
    -----------
    v : float
        Degrees of freedom (v > 0)
    logsigma : torch.tensor 
        Diagonal of the log covariance matrix of shape (d,)
        
    Returns:
    --------
    torch.Tensor
        The computed Ψ values
    """
    # Compute log numerator: log(Γ((v+k)/2))
    log_numerator = math.lgamma((v + k) / 2)
    
    # Compute log denominator parts
    log_pi_v_term = math.log(math.pi * v) * (k/2)
    log_gamma_term = math.lgamma(v/2)
    #log_det_term = torch.sum(log_sigma)*(1/2)
    
    # Combine log terms
    log_base = log_numerator - log_pi_v_term - log_gamma_term #- log_det_term
    
    # Compute final result in log space and then exponentiate
    log_psi = log_base * (-2/(v + k))
    psi = math.exp(log_psi)
    
    return torch.tensor(psi, dtype=torch.float32)

def compute_t_st(mu1, log_sigma1, mu2, log_sigma2, q, v, sum = True, lamb = 1, initial_prior_var = 1):
    """
    Calculate the element-wise t-divergence between two d-dimensional distributions given the dof v.
    
    Parameters:
    -----------
    mu1 : torch.Tensor
        Mean of the first distribution, shape (d,)
    mu2 : torch.Tensor
        Mean of the second distribution, shape (d,)
    log_sigma1 : torch.Tensor
        Log variance of the first distribution, shape (d,)
    log_sigma2 : torch.Tensor
        Log variance of the second distribution, shape (d,)
    v : int or torch.Tensor
        Degrees of freedom (v > 0). Keep between values of 1.0 and 50.0 (the larger it gets, t converges to 1 as in the Gaussian)
        
    Returns:
    --------
    torch.Tensor
        The element-wise t-divergence values, shape (d,)
    """
    
    # Check if v is positive
    if v <= 0:
        raise ValueError("Degrees of freedom v must be positive")
    
    # Ensure all inputs have the same shape
    assert mu1.shape == mu2.shape == log_sigma1.shape == log_sigma2.shape, "All inputs must have the same shape"
    
    k = mu1.shape[0]
    # Calculate t from v: t = 2/(v+1) + 1
    t = 2/(v + k) + 1
    den = 1 - t  # = -2/(v+1) 

    #Calcualte dimensionality
    k = mu1.shape[0]
    
    # Compute Ψ₁ and Ψ₂ (now returns tensors of shape (d,))
    psi = compute_psi_vectorized_ln(v, k)/abs(den) #note abs of den is taken because after everything is negated for numerical stability
    #implement ignoring the gamma terms, as they will be the same for all divergences (just a constant multiplier)
    #psi1 = torch.prod(torch.exp(log_sigma1/(v+k))) #computes the determinant od a diagonal matrix raised to the power of 1/v+k
    #psi2 = torch.prod(torch.exp(log_sigma2/(v+k)))
    #psi1 = torch.exp(torch.sum(log_sigma1)/(v+k))
    #psi2 = torch.exp(torch.sum(log_sigma2)/(v+k))
    
    # Common denominator terms: for t larger than 1 (the case that we are considering, den is negative)

    det_term1 = torch.exp(torch.sum(log_sigma1)/(v+k))
    det_term2 = torch.exp(torch.sum(log_sigma2)/(v+k))
    #mean_term = (mu1 - mu2)**2 * torch.exp(-log_sigma2)/v
    mean_term = (mu1 - mu2)**2 * torch.exp(-log_sigma2)/v
    trace_term = torch.exp(log_sigma1 - log_sigma2)/v
    #trace_term = torch.exp(log_sigma1 - log_sigma2)/v
    det_term = torch.exp(log_sigma1/(v+k)) - torch.exp(log_sigma2/(v+k))
    if sum:
        #divergence = torch.pow(torch.sum(torch.pow((term1 + term2 + term3 + term4 + term5),den)) - (dim - 1), 1/den)
        divergence = psi*torch.sum(-det_term + det_term2*mean_term + det_term2*trace_term) - psi*det_term1/v
    else:
        divergence =  psi*(det_term2*mean_term + det_term2*trace_term -det_term - det_term1/v)
    
    return divergence
def compute_t_st_mf(mu1, log_sigma1, mu2, log_sigma2, q, v, sum = True, lamb = 1, initial_prior_var = 1):
    """
    Calculate the element-wise t-divergence between two d-dimensional distributions given q (which equals t)
    
    Note  t = 2/(v+k) + 1 and v = 2/(t-1) - k

    Take mean field approach and assume v_new  = v -k +1. One obtains v = 2/(t-1) - 1

    Parameters:
    -----------
    mu1 : torch.Tensor
        Mean of the first distribution, shape (d,)
    mu2 : torch.Tensor
        Mean of the second distribution, shape (d,)
    log_sigma1 : torch.Tensor
        Log variance of the first distribution, shape (d,)
    log_sigma2 : torch.Tensor
        Log variance of the second distribution, shape (d,)
    v : int or torch.Tensor
        Degrees of freedom (v > 0). Keep between values of 1.0 and 50.0 (the larger it gets, t converges to 1 as in the Gaussian)
        
    Returns:
    --------
    torch.Tensor
        The element-wise t-divergence values, shape (d,)
    """
    
    # Check if v is positive
    if v <= 0:
        raise ValueError("Degrees of freedom v must be positive")
    
    # Ensure all inputs have the same shape
    assert mu1.shape == mu2.shape == log_sigma1.shape == log_sigma2.shape, "All inputs must have the same shape"
    # Calculate t from v: t = 2/(v+1) + 1
    dof = 2/(q-1) - 1 #degree of fredom of the univariate t distribution based on q value
    den = 1 - q  # = -2/(v+1) 

    # Compute Ψ₁ and Ψ₂ (now returns tensors of shape (d,))
    psi = compute_psi_vectorized_ln(dof, 1)/abs(den) #note abs of den is taken because after everything is negated for numerical stability

    det_term1 = torch.exp(torch.sum(log_sigma1)/(dof+1))
    det_term2 = torch.exp(torch.sum(log_sigma2)/(dof+1))

    mean_term = (mu1 - mu2)**2 * torch.exp(-log_sigma2)/dof
    trace_term = torch.exp(log_sigma1 - log_sigma2)/dof

    det_term = torch.exp(log_sigma1/(dof+1)) - torch.exp(log_sigma2/(dof+1))
    if sum:
        divergence = psi*torch.sum(-det_term + det_term2*mean_term + det_term2*trace_term) - psi*det_term1/dof
    else:
        divergence =  psi*(det_term2*mean_term + det_term2*trace_term -det_term - det_term1/dof)
    
    return divergence



def compute_t_st_old(mu1, log_sigma1, mu2, log_sigma2, q, v, sum = True, lamb = 1, initial_prior_var = 1):
    """
    Calculate the element-wise t-divergence between two d-dimensional distributions.
    
    Parameters:
    -----------
    mu1 : torch.Tensor
        Mean of the first distribution, shape (d,)
    mu2 : torch.Tensor
        Mean of the second distribution, shape (d,)
    log_sigma1 : torch.Tensor
        Log variance of the first distribution, shape (d,)
    log_sigma2 : torch.Tensor
        Log variance of the second distribution, shape (d,)
    v : int or torch.Tensor
        Degrees of freedom (v > 0). Keep between values of 1.0 and 50.0 (the larger it gets, t converges to 1 as in the Gaussian)
        
    Returns:
    --------
    torch.Tensor
        The element-wise t-divergence values, shape (d,)
    """
    
    # Check if v is positive
    if v <= 0:
        raise ValueError("Degrees of freedom v must be positive")
    
    # Ensure all inputs have the same shape
    assert mu1.shape == mu2.shape == log_sigma1.shape == log_sigma2.shape, "All inputs must have the same shape"
    
    # Calculate t from v: t = 2/(v+1) + 1
    t = 2/(v + 1) + 1

    #Calcualte dimensionality
    k = mu1.shape[0]
    
    # Compute Ψ₁ and Ψ₂ (now returns tensors of shape (d,))
    #psi1 = compute_psi_vectorized(v, log_sigma1)
    #psi2 = compute_psi_vectorized(v, log_sigma2)
    #implement ignoring the gamma terms, as they will be the same for all divergences (just a constant multiplier)
    psi1 = torch.prod(torch.exp(log_sigma1/(v+k))) #computes the determinant od a diagonal matrix raised to the power of 1/v+k
    psi2 = torch.prod(torch.exp(log_sigma2/(v+k)))
    
    # Common denominator terms
    den = 1 - t  # = -2/(v+1)
    
    # Convert log_sigma2 to sigma2 for calculations
    sigma1 = torch.exp(log_sigma1)
    sigma2 = torch.exp(log_sigma2)
    v_sigma2 = v * sigma2
    
    # Calculate each term (element-wise operations)
    term1 = (psi1/den) * (1 + 1/v)
    term2 = 2 * (mu1 * mu2/v_sigma2)
    term3 = -1 * (sigma1/v_sigma2)
    term4 = -1 * (mu1 * mu1/v_sigma2)
    term5 = -1 * (mu2 * mu2/v_sigma2 + 1)
    
    if sum:
        #divergence = torch.pow(torch.sum(torch.pow((term1 + term2 + term3 + term4 + term5),den)) - (dim - 1), 1/den)
        divergence = torch.sum((psi1/den)*term1 + (psi2/den)*(term2 + term3 + term4 + term5))
    else:
        divergence = (psi1/den)*term1 + (psi2/den)*(term2 + term3 + term4 + term5)
    
    return divergence

def compute_kl_qg(mean, log_var, prior_mean, log_prior_var, alpha = 2, sum = True, lamb = 1, initial_prior_var = 1):
    '''
    KL divergence between two q-Gaussians. To be implemented
    '''
    return 1

def compute_re_qg(mean, log_var, prior_mean, log_prior_var, alpha = 2, sum = True, lamb = 1, initial_prior_var = 1):
    '''
    t-Divergence between two Student-t distributions with diagonal covariances
    '''
    
    return 1
    
def compute_kl_true(mean, exp_var, prior_mean, prior_exp_var, alpha = 2, sum = True, lamb = 1, initial_prior_var = 1):

    '''
    Computes the Dkl(approximate distribution || prior distribution) between two Gaussian distributions

    Note that the variances are log variances
    '''

    #currently passing the variance of each individual weight, pass into a covariance form
    cov_matrix = torch.diag_embed(torch.exp(exp_var))
    prior_cov_matrix = torch.diag_embed(torch.exp(prior_exp_var))


    dist1 = dist.MultivariateNormal(mean, cov_matrix)
    dist2 = dist.MultivariateNormal(prior_mean, prior_cov_matrix)
    kl_loss = torch.distributions.kl.kl_divergence(dist1, dist2)
    return kl_loss

def sample_student_t(shape, df, device="cuda", dtype=torch.float32):
    """
    Efficiently sample from a Student's t-distribution using PyTorch.
    
    Parameters:
        shape (tuple): The shape of the output samples.
        df (float): Degrees of freedom of the Student's t-distribution.
        loc (float): Mean (location parameter) of the distribution.
        scale (float): Scale parameter of the distribution.
        device (str): The device to use ('cpu' or 'cuda').
        dtype (torch.dtype): The data type for the samples.

    Returns:
        torch.Tensor: Samples from the Student's t-distribution.
    """
    # Generate standard normal samples
    X = torch.empty(shape, device=device).normal_(mean=0,std=1)
    
    # Generate chi-squared samples
    Z = torch.distributions.Chi2(df).sample(shape).to(device=device)
    
    # Transform to Student's t-distribution
    Y = X * torch.rsqrt(Z / df)
    
    # Apply location and scale
    return Y
