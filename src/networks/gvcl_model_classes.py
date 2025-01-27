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

from . import compute_kl_g, compute_re_g, compute_t_st, compute_t_st_k1, compute_t_st_mf, sample_student_t


device = 'cuda:0'
class MultiHeadCNN(nn.Module):
    """Multihead CNN without FiLM"""
    def __init__(self, input_shape, conv_sizes, fc_sizes, output_dims, single_head = False, global_avg_pool = False, prior_var = 1, init_vars = [], activation_fun="relu"):
        super().__init__()
        self.conv_layers = nn.ModuleList([])
        self.fc_layers = nn.ModuleList([])
        self.heads = nn.ModuleList([])
        self.num_tasks = len(output_dims)
        self.output_dims = output_dims

        self.single_head = single_head
        self.prior_var = prior_var
        self.global_avg_pool = global_avg_pool
        self.pool_indices = []        

        self.prior_var = prior_var

        if activation_fun == "relu":
            self.act=F.relu
        elif activation_fun == "tanh":
            self.act=F.tanh
        elif activation_fun == "sigmoid": 
            self.act=F.sigmoid
        elif activation_fun == "leaky_relu":
            self.act=F.leaky_relu
        else:
            raise ValueError(f"Unknown activation function: {activation_fun}")

        if len(init_vars) == 0: init_vars = -7 * np.ones([len(conv_sizes) + len(fc_sizes) + 1])
        
        layer_index = 0

        channels = input_shape[0]
        prev_channels = input_shape[0]
        input_dimension = input_shape[1]
        for i, layer_params in enumerate(conv_sizes):
            if layer_params == 'pool':
                self.pool_indices.append(i - len(self.pool_indices) - 1)
                input_dimension = input_dimension//2
                continue
            channels = layer_params[0]
            kernel_size = layer_params[1]
            if(len(layer_params) == 2):
                padding = int((kernel_size-1)/2)
            else:
                padding = layer_params[2]
            conv_layer = MFConvLayer(prev_channels, channels, kernel_size=kernel_size, stride=1, padding=padding, prior_var = self.prior_var, init_var = init_vars[layer_index])
            
            input_dimension = int(np.floor((input_dimension+2*padding-(kernel_size-1)-1)/float(1)+1))
            prev_channels = channels
            self.conv_layers.append(conv_layer)
            layer_index += 1

        last_size = channels * input_dimension**2
        
        if global_avg_pool:
            last_size = channels
        
        for i, hidden_size in enumerate(fc_sizes):
            self.fc_layers.append(MFLinearLayer(last_size, hidden_size, prior_var = self.prior_var, init_var = init_vars[layer_index]))
            last_size = hidden_size
            layer_index += 1
        
        if single_head:
            self.heads.append(MFLinearLayer(last_size, output_dims[0], prior_var = self.prior_var, init_var = init_vars[layer_index]))
        else:
            for output_dim in output_dims:
                self.heads.append(MFLinearLayer(last_size, output_dim, prior_var = self.prior_var, init_var = init_vars[layer_index]))


    def get_task_specific_parameters(self, task_number):
        modules = nn.ModuleList([])
        if not self.single_head:
            modules.append(self.heads[task_number])
        modules.append(self.fc_layers)
        modules.append(self.conv_layers)
        if self.single_head:
            modules.append(self.heads[0])
        
        return modules.parameters()
    
    def get_all_parameters(self, task_number, prior=False):
        all_params = []
        if prior:
            all_params.extend(self.heads[task_number].get_prior_params())
            for layer in self.fc_layers:
                all_params.extend(layer.get_prior_params())
            for layer in self.conv_layers:
                all_params.extend(layer.get_prior_params())
        else: 
            all_params.extend(self.heads[task_number].get_current_params())
            for layer in self.fc_layers:
                all_params.extend(layer.get_current_params())
            for layer in self.conv_layers:
                all_params.extend(layer.get_current_params())
        return all_params

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, x, task_labels, reg_type, v, num_samples=1, tasks = None):
        if tasks is None:
            tasks = range(self.num_tasks)
            excluded_tasks = []
        else:
            excluded_tasks = [i for i in range(self.num_tasks) if i not in tasks]

        outputs = [None for j in range(self.num_tasks)]

        batch_size = x.shape[0]
        if 't_st'in reg_type:x = x.repeat([num_samples,1,1,1,1])
        else:x = x.repeat([num_samples,1,1,1])
        
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x,reg_type,v, num_samples)  
            x = self.act(x) 
            if i in self.pool_indices:
                if 't_st'in reg_type:x = x.view(-1, *x.shape[2:])
                x = F.max_pool2d(x, kernel_size = 2, stride = 2)
                if 't_st'in reg_type:x = x.view(num_samples, batch_size, *x.shape[1:])
        
        if self.global_avg_pool:
            x = x.view(num_samples, batch_size, x.shape[1], -1).mean(-1)
        else:
            x = x.view(num_samples, batch_size, -1)
        
        for i, layer in enumerate(self.fc_layers):
            x = layer(x, reg_type,v) 
            x = self.act(x)
            
        self.pre_head = x

        for j in tasks:
            head_index = 0 if self.single_head else j
            task_output = self.heads[head_index](x,reg_type,v)
            outputs[j] = task_output.reshape([num_samples, batch_size, -1])
        for j in excluded_tasks:
            outputs[j] = torch.zeros_like(task_output, device = device)

        return outputs
    
    
    def collect_all_variances_vector(self, task, prior=False):
        all_vars = []
        for layer in self.fc_layers:
            all_vars.append(layer.get_var(prior))
        for layer in self.conv_layers:
            all_vars.append(layer.get_var(prior))
        all_vars.append(self.heads[task].get_var(prior))
        return torch.cat([torch.flatten(var) for var_pair in all_vars for var in var_pair])


    def set_prior_grads(self, flag):
        for layer in self.fc_layers:
            layer.set_prior_grads(flag)
        for layer in self.conv_layers:
            layer.set_prior_grads(flag)

    def forward_mean(self, x, task_labels, reg_type, v, tasks = None, prior=False):
        if tasks is None:
            tasks = range(self.num_tasks)
            excluded_tasks = []
        else:
            excluded_tasks = [i for i in range(self.num_tasks) if i not in tasks]

        outputs = [None for j in range(self.num_tasks)]

        batch_size = x.shape[0]
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer.forward_mean(x,reg_type,v, prior=prior)  
            x = self.act(x) 
            if i in self.pool_indices:
                x = F.max_pool2d(x, kernel_size = 2, stride = 2)
        
        if self.global_avg_pool:
            x = x.view(batch_size, x.shape[1], -1).mean(-1)
        else:
            x = x.view(batch_size, -1)
        
        for i, layer in enumerate(self.fc_layers):
            x = layer.forward_mean(x, reg_type,v, prior=prior) 
            x = self.act(x)
            
        self.pre_head = x

        for j in tasks:
            head_index = 0 if self.single_head else j
            task_output = self.heads[head_index].forward_mean(x,reg_type,v, prior=prior)
            outputs[j] = task_output.reshape([batch_size, -1])
        for j in excluded_tasks:
            outputs[j] = torch.zeros_like(task_output, device = device)

        return outputs

    def get_reg(self, lamb, reg_type, q,v):
        kl = 0

        for i, conv_layer in enumerate(self.conv_layers):
            kl += conv_layer.get_reg(lamb, reg_type, q, v)
        
        for layer in self.fc_layers:
            kl += layer.get_reg(lamb, reg_type, q, v)

        for t, layer in enumerate(self.heads):
            kl += layer.get_reg(lamb, reg_type, q, v)

        return kl
    
    def add_task_body_params(self, updated_tasks,keep_grad_mean=False):
        for layer in self.fc_layers:
            layer.add_new_task(keep_grad_mean = keep_grad_mean)
        for layer in self.conv_layers:
            layer.add_new_task(keep_grad_mean = keep_grad_mean)
        if self.single_head:
            self.heads[0].add_new_task(keep_grad_mean=keep_grad_mean, reset_variance = False)
        if not self.single_head:
            for t in updated_tasks:
                self.heads[t].add_new_task(keep_grad_mean=keep_grad_mean, reset_variance = False)


class MultiHeadFiLMCNN(nn.Module):
    def __init__(self, input_shape, conv_sizes, fc_sizes, output_dims, film_type = 'point', single_head = False, global_avg_pool = False, prior_var = 1, init_vars = []):
        super().__init__()
        self.conv_layers = nn.ModuleList([])
        self.fc_layers = nn.ModuleList([])
        self.heads = nn.ModuleList([])
        self.num_tasks = len(output_dims)
        self.output_dims = output_dims
        self.film_type = film_type

        self.single_head = single_head
        self.prior_var = prior_var
        self.set_film_gen_type()
        self.global_avg_pool = global_avg_pool
        self.pool_indices = []        
        print("Film type", self.film_type)

        self.conv_film_layers = nn.ModuleList([self.conv_film_gen_type(self.num_tasks, conv_size[0]) for conv_size in conv_sizes if conv_size != 'pool'])
        self.fc_film_layers = nn.ModuleList([self.fc_film_gen_type(self.num_tasks, fc_size) for fc_size in fc_sizes])

        self.prior_var = prior_var

        if len(init_vars) == 0:
            init_vars = -7 * np.ones([len(conv_sizes) + len(fc_sizes) + 1])
        
        layer_index = 0

        channels = input_shape[0]
        prev_channels = input_shape[0]
        input_dimension = input_shape[1]
        for i, layer_params in enumerate(conv_sizes):
            if layer_params == 'pool':
                self.pool_indices.append(i - len(self.pool_indices) - 1)
                input_dimension = input_dimension//2
                continue
            channels = layer_params[0]
            kernel_size = layer_params[1]
            if(len(layer_params) == 2):
                padding = int((kernel_size-1)/2)
            else:
                padding = layer_params[2]
            conv_layer = MFConvLayer(prev_channels, channels, kernel_size=kernel_size, stride=1, padding=padding, prior_var = self.prior_var, init_var = init_vars[layer_index])
            
            input_dimension = int(np.floor((input_dimension+2*padding-(kernel_size-1)-1)/float(1)+1))
            prev_channels = channels
            self.conv_layers.append(conv_layer)
            layer_index += 1

        last_size = channels * input_dimension**2
        
        if global_avg_pool:
            last_size = channels
        
        for i, hidden_size in enumerate(fc_sizes):
            self.fc_layers.append(MFLinearLayer(last_size, hidden_size, prior_var = self.prior_var, init_var = init_vars[layer_index]))
            last_size = hidden_size
            layer_index += 1

        last_size_copy = last_size
        for task in range(self.num_tasks):
            last_size_copy = last_size
            layers = nn.ModuleList([])
        last_size = last_size_copy
        
        if single_head:
            self.heads.append(MFLinearLayer(last_size, output_dims[0], prior_var = self.prior_var, init_var = init_vars[layer_index]))
        else:
            for output_dim in output_dims:
                self.heads.append(MFLinearLayer(last_size, output_dim, prior_var = self.prior_var, init_var = init_vars[layer_index]))

    def get_film_type(self):
        return self.film_type

    def get_task_specific_parameters(self, task_number):
        modules = nn.ModuleList([self.conv_film_layers, self.fc_film_layers])
        if not self.single_head:
            modules.append(self.heads[task_number])
        modules.append(self.fc_layers)
        modules.append(self.conv_layers)
        if self.single_head:
            modules.append(self.heads[0])
        
        return modules.parameters()

    def set_film_gen_type(self):
        if(self.film_type == 'point'):
            self.conv_film_gen_type = partial(PointFiLMLayer, constant = False, conv = True)
            self.fc_film_gen_type = partial(PointFiLMLayer, constant = False)
        elif(self.film_type == 'scale'):
            self.conv_film_gen_type = partial(PointFiLMLayer, constant = False, conv = True, scale_only = True)
            self.fc_film_gen_type = partial(PointFiLMLayer, constant = False, scale_only = True)
        elif(self.film_type == 'bias'):
            self.conv_film_gen_type = partial(PointFiLMLayer, constant = False, conv = True, bias_only = True)
            self.fc_film_gen_type = partial(PointFiLMLayer, constant = False, bias_only = True)
        elif(self.film_type == 'none'):
            self.conv_film_gen_type = partial(PointFiLMLayer, constant = True, conv = True)
            self.fc_film_gen_type = partial(PointFiLMLayer, constant = True)

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, x, task_labels, reg_type, v, num_samples=1, tasks = None):
        if tasks is None:
            tasks = range(self.num_tasks)
            excluded_tasks = []
        else:
            excluded_tasks = [i for i in range(self.num_tasks) if i not in tasks]

        num_total_tasks = len(tasks)

        outputs = [None for j in range(self.num_tasks)]

        batch_size = x.shape[0]
        if 't_st'in reg_type:x = x.repeat([num_samples,1,1,1,1])
        else:x = x.repeat([num_samples,1,1,1])
        
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x,reg_type,v, num_samples) 
            if not self.film_type == 'none': # excluding the film layer from the forward pass when 'none'
                x = self.conv_film_layers[i](x, task_labels, num_samples)
            
            x = F.relu(x) 
            if i in self.pool_indices:
                if 't_st'in reg_type:x = x.view(-1, *x.shape[2:])
                x = F.max_pool2d(x, kernel_size = 2, stride = 2)
                if 't_st'in reg_type:x = x.view(num_samples, batch_size, *x.shape[1:])
        
        if self.global_avg_pool:
            x = x.view(num_samples, batch_size, x.shape[1], -1).mean(-1)
        else:
            x = x.view(num_samples, batch_size, -1)
        
        for i, layer in enumerate(self.fc_layers):
            x = layer(x, reg_type,v)
            if not self.film_type == 'none': # excluding the film layer from the forward pass when 'none'
                x = self.fc_film_layers[i](x, task_labels, num_samples)
            
            x = F.relu(x)
            
        self.pre_head = x

        for j in tasks:
            head_index = 0 if self.single_head else j
            task_output = self.heads[head_index](x,reg_type,v)
            outputs[j] = task_output.reshape([num_samples, batch_size, -1])
        for j in excluded_tasks:
            outputs[j] = torch.zeros_like(task_output, device = device)

        return outputs

    def get_reg(self, lamb, reg_type, q,v):
        kl = 0

        for i, conv_layer in enumerate(self.conv_layers):
            kl += conv_layer.get_reg(lamb, reg_type, q, v)
        
        for layer in self.fc_layers:
            kl += layer.get_reg(lamb, reg_type, q, v)

        for t, layer in enumerate(self.heads):
            kl += layer.get_reg(lamb, reg_type, q, v)

        return kl
    
    def add_task_body_params(self, updated_tasks):
        for layer in self.fc_layers:
            layer.add_new_task()
        for layer in self.conv_layers:
            layer.add_new_task()
        if self.single_head:
            self.heads[0].add_new_task(reset_variance = False)
        if not self.single_head:
            for t in updated_tasks:
                self.heads[t].add_new_task(reset_variance = False)

class PointFiLMLayer(nn.Module):
    def __init__(self, tasks, width, constant = False, conv = False, scale_only = False, bias_only = False):
        super().__init__()
        self.scales = Parameter(torch.Tensor(tasks, width), requires_grad = (not constant) and (not bias_only))
        self.shifts = Parameter(torch.Tensor(tasks, width), requires_grad = (not constant) and (not scale_only))
        self.conv = conv
        self.width = width
        self.constant = constant
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.scales, 1.)
        init.constant_(self.shifts, 0.)
    
    def forward(self, x, task_labels, num_samples):
        scale_values = F.embedding(task_labels, self.scales)
        shift_values = F.embedding(task_labels, self.shifts)

        if self.conv:
            scale_values = scale_values.view(-1, self.width, 1, 1).repeat(num_samples, 1,1,1)
            shift_values = shift_values.view(-1, self.width, 1, 1).repeat(num_samples, 1,1,1)
        else:
            scale_values = scale_values.view(1, -1, self.width)
            shift_values = shift_values.view(1, -1, self.width)

        return x * scale_values + shift_values

class MFConvLayer(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', prior_var = 1, init_var = -7):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        
        self.init_var = init_var
        
        self.W_prior_mean = torch.zeros(self.weight.shape, device = device)
        self.b_prior_mean = torch.zeros(self.bias.shape, device = device)
        
        self.prior_var = prior_var
        self.W_prior_var = torch.ones(self.weight.shape, device = device).mul(np.log(self.prior_var))
        self.b_prior_var = torch.ones(self.bias.shape, device = device).mul(np.log(self.prior_var))

        self.weight_var = Parameter(torch.Tensor(self.weight.shape))
        self.bias_var = Parameter(torch.Tensor(self.bias.shape))

        
        self.reset_parameters()

    def get_prior_params(self):
        return [self.W_prior_mean, self.b_prior_mean]

    def get_current_params(self):
        return [self.weight, self.bias]
    
    def conv2d_forward(self, input, weight, bias):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'weight_var'):
            init.constant_(self.weight_var, self.init_var)
            init.constant_(self.bias_var, self.init_var)

    def add_new_task(self):
        self.W_prior_mean = self.weight.clone().detach().requires_grad_(False)
        self.b_prior_mean = self.bias.clone().detach().requires_grad_(False)
        
        self.W_prior_var = self.weight_var.clone().detach().requires_grad_(False)
        self.b_prior_var = self.bias_var.clone().detach().requires_grad_(False)
        
        self.weight_var.data = torch.min(self.weight_var, self.init_var*torch.ones_like(self.weight_var).data)
        self.bias_var.data = torch.min(self.bias_var, self.init_var*torch.ones_like(self.bias_var).data)

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)

        initialization_noise = torch.empty_like(self.weight)
        init.kaiming_uniform_(initialization_noise, a = math.sqrt(5))
        # self.weight.data = self.weight.data + (self.weight_var > -2).float() * initialization_noise
        # self.bias.data = self.bias.data + (self.bias_var > -2).float() * torch.empty_like(self.bias).uniform_(-bound, bound)

        self.weight.data = initialization_noise.data
        self.bias.data = torch.empty_like(self.bias).uniform_(-bound, bound).data

    def get_var(self, prior=False):
        if prior:
            return [torch.exp(self.W_prior_var), torch.exp(self.b_prior_var)]
        else:
            return [torch.exp(self.weight_var), torch.exp(self.bias_var)]
        
    def get_kl(self, lamb):

        W_kl = compute_kl_g(self.weight, self.weight_var, self.W_prior_mean, self.W_prior_var, lamb = lamb, initial_prior_var = self.prior_var)
        b_kl = compute_kl_g(self.bias, self.bias_var, self.b_prior_mean, self.b_prior_var, lamb = lamb, initial_prior_var = self.prior_var)

        return W_kl + b_kl

    def get_reg(self, lamb, reg_type, q, v):
        '''
        Function that computes either of the possible 4 regularization types;
        1. KL between two Gaussians
        2. Renyi of degree q between two Gaussians
        3. KL between two q-gaussians (of order q)
        4. Renyi between two q-gaussians (of order q)

        Returns:
        divergence value (torch.float)
        '''
        if reg_type == 'kl_g':
            kl_function = compute_kl_g
        elif reg_type == 're_g':
            kl_function = compute_re_g
        elif reg_type == "kl_qg":
            kl_function = compute_kl_qg
        elif reg_type == "t_st":
            kl_function = compute_t_st
        elif reg_type == "t_st_mf":
            kl_function = compute_t_st_mf
    
        else:
            raise ValueError(f"Unknown regularization type: {reg_type}")

        W_kl = kl_function(self.weight, self.weight_var, self.W_prior_mean, self.W_prior_var, q, v, lamb=lamb, initial_prior_var=self.prior_var)
        b_kl = kl_function(self.bias, self.bias_var, self.b_prior_mean, self.b_prior_var, q, v, lamb=lamb, initial_prior_var=self.prior_var)
        return W_kl + b_kl


    def forward_t_st(self, input, v, num_samples=1):

        outputs = []
        for i in range(num_samples):
            W_eps = sample_student_t(self.weight.shape, v, device=device)
            W_sigma = torch.sqrt(torch.exp(self.weight_var))
            weight = self.weight + W_eps * W_sigma

            bias_eps = sample_student_t(self.bias.shape, v, device=device)
            bias_sigma = torch.sqrt(torch.exp(self.bias_var))
            bias = self.bias + bias_eps * bias_sigma

            outputs.append(self.conv2d_forward(input[i], weight, bias))

        return torch.stack(outputs)


    def forward_t_st_prior(self, input, v, num_samples=1):

        outputs = []
        for i in range(num_samples):
            W_eps = sample_student_t(self.weight.shape, v, device=device)
            W_sigma = torch.sqrt(torch.exp(self.W_prior_var))
            weight = self.W_prior_mean + W_eps * W_sigma

            bias_eps = sample_student_t(self.bias.shape, v, device=device)
            bias_sigma = torch.sqrt(torch.exp(self.b_prior_mean))
            bias = self.b_prior_var + bias_eps * bias_sigma

            outputs.append(self.conv2d_forward(input[i], weight, bias))

        return torch.stack(outputs)

    def forward(self, input, reg_type, v, num_samples=-1):
        if 't_st'in reg_type:
            return self.forward_t_st(input, v, num_samples)
        
        output_mean =  self.conv2d_forward(input, self.weight, self.bias)
        output_var = self.conv2d_forward(input**2, torch.exp(self.weight_var), torch.exp(self.bias_var))

        if 't_st'in reg_type:
            #print("Student-t sampling")
            eps = sample_student_t(output_mean.shape, v, device=device)
        else:
            #print("Normal sampling")
            eps = torch.empty(output_mean.shape, device=device).normal_(mean=0,std=1)
        output = output_mean + torch.sqrt(output_var + 1e-9) * eps

        return output


    def forward_mean(self, input, reg_type, v, num_samples=-1, prior=False):
        if not prior:
            return self.conv2d_forward(input, self.weight, self.bias)

        return  self.conv2d_forward(input, self.W_prior_mean, self.b_prior_mean)

class MFLinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, prior_var = 1, init_var = -7):
        super().__init__()
        self.init_var = init_var
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.W_mean = Parameter(torch.Tensor(dim_out, dim_in))
        self.b_mean = Parameter(torch.Tensor(dim_out))

        self.W_var = Parameter(torch.Tensor(dim_out, dim_in))
        self.b_var = Parameter(torch.Tensor(dim_out))

        self.W_prior_mean = torch.zeros([dim_out, dim_in], device=device)
        self.b_prior_mean = torch.zeros([dim_out], device=device)

        self.prior_var = prior_var
        
        self.W_prior_var = torch.ones([dim_out, dim_in], device = device).mul(np.log(self.prior_var))
        self.b_prior_var = torch.ones([dim_out], device = device).mul(np.log(self.prior_var))


        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.W_mean, a=math.sqrt(5))

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.W_mean)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b_mean, -bound, bound)

        init.constant_(self.W_var, self.init_var)
        init.constant_(self.b_var, self.init_var)


    def get_prior_params(self):
        return [self.W_prior_mean, self.b_prior_mean]
    

    def get_current_params(self):
        return [self.W_mean, self.b_mean]
    
    def add_new_task(self, reset_variance = True, keep_grad_mean=False):
        self.W_prior_mean = self.W_mean.clone().detach().requires_grad_(False)
        self.b_prior_mean = self.b_mean.clone().detach().requires_grad_(False)
    
        self.W_prior_var = self.W_var.clone().detach().requires_grad_(False)
        self.b_prior_var = self.b_var.clone().detach().requires_grad_(False)
        
        if reset_variance:
            self.W_var.data = torch.min(self.W_var, self.init_var*torch.ones_like(self.W_var).data)
            self.b_var.data = torch.min(self.b_var, self.init_var*torch.ones_like(self.b_var).data)

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W_mean)
            bound = 1 / math.sqrt(fan_in)

            initialization_noise = torch.empty_like(self.W_mean)
            init.kaiming_uniform_(initialization_noise, a = math.sqrt(5))
            # self.W_mean.data = self.W_mean.data + (self.W_var > -2).float() * initialization_noise
            # self.b_mean.data = self.b_mean.data + (self.b_var > -2).float() * torch.empty_like(self.b_mean).uniform_(-bound, bound)

            self.W_mean.data = initialization_noise.data
            self.b_mean.data = torch.empty_like(self.b_mean).uniform_(-bound, bound).data

    def get_kl(self, lamb):
        W_kl = compute_kl_g(self.W_mean, self.W_var, self.W_prior_mean, self.W_prior_var, lamb = lamb, initial_prior_var = self.prior_var)
        b_kl = compute_kl_g(self.b_mean, self.b_var, self.b_prior_mean, self.b_prior_var, lamb = lamb, initial_prior_var = self.prior_var)
        return W_kl + b_kl

    def get_reg(self, lamb, reg_type, q, v):
        '''
        Function that computes either of the possible 4 regularization types;
        1. KL between two Gaussians
        2. Renyi of degree q between two Gaussians
        3. KL between two q-gaussians (of order q)
        4. Renyi between two q-gaussians (of order q)

        Returns:
        divergence value (torch.float)
        '''
        if reg_type == 'kl_g':
            kl_function = compute_kl_g
        elif reg_type == 're_g':
            kl_function = compute_re_g
        elif reg_type == "kl_qg":
            kl_function = compute_kl_qg
        elif reg_type == "t_st":
            kl_function = compute_t_st
        elif reg_type == "t_st_mf":
            kl_function = compute_t_st_mf
        else:
            raise ValueError(f"Unknown regularization type: {reg_type}")

        W_kl = kl_function(self.W_mean, self.W_var, self.W_prior_mean, self.W_prior_var, q, v, lamb=lamb, initial_prior_var=self.prior_var)
        b_kl = kl_function(self.b_mean, self.b_var, self.b_prior_mean, self.b_prior_var, q, v, lamb=lamb, initial_prior_var=self.prior_var)
        return W_kl + b_kl

    def forward(self, x, reg_type, v):
        output_mean = x.matmul(self.W_mean.t()) + self.b_mean.unsqueeze(0).unsqueeze(0)
        output_std = torch.sqrt((x**2).matmul(torch.exp(self.W_var.t())) + torch.exp(self.b_var).unsqueeze(0).unsqueeze(0))
        if  't_st'in reg_type:
            #print("Student-t sampling")
            eps = sample_student_t(output_mean.shape, v, device=device)
        else:
            #print("Normal sampling")
            eps = torch.empty(output_mean.shape, device=device).normal_(mean=0,std=1)

        output = output_mean + (eps * output_std)
        return output

'''
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
        return 0.5 * mahalanobis_term - 0.5 / (alpha - 1)*log_det_term

#def compute_psi_vectorized_ln(v, k):
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
'''