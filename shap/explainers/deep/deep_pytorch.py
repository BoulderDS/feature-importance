import numpy as np
import warnings
from shap.explainers.explainer import Explainer
from tqdm import tqdm
from distutils.version import LooseVersion
torch = None


class PyTorchDeepExplainer(Explainer):

    def __init__(self, model, data):
        # try and import pytorch
        global torch
        if torch is None:
            import torch
            if LooseVersion(torch.__version__) < LooseVersion("0.4"):
                warnings.warn("Your PyTorch version is older than 0.4 and not supported.")

        # check if we have multiple inputs
        self.multi_input = False
        if type(data) == list:
            self.multi_input = True
        if type(data) != list:
            data = [data]
        self.data = data
        self.layer = None
        self.input_handle = None
        self.interim = False
        self.interim_inputs_shape = None
        self.expected_value = None  # to keep the DeepExplainer base happy
        if type(model) == tuple:
            self.interim = True
            model, layer = model
            model = model.eval()
            self.layer = layer
            self.add_target_handle(self.layer)

            # if we are taking an interim layer, the 'data' is going to be the input
            # of the interim layer; we will capture this using a forward hook
            with torch.no_grad():
                _ = model(*data)
                interim_inputs = self.layer.target_input
                if type(interim_inputs) is tuple:
                    # this should always be true, but just to be safe
                    self.interim_inputs_shape = [i.shape for i in interim_inputs]
                else:
                    self.interim_inputs_shape = [interim_inputs.shape]
            self.target_handle.remove()
            del self.layer.target_input
        self.model = model.eval()

        self.multi_output = False
        self.num_outputs = 1
        with torch.no_grad():
            outputs = model(*data)
            if outputs.shape[1] > 1:
                self.multi_output = True
                self.num_outputs = outputs.shape[1]

    def add_target_handle(self, layer):
        input_handle = layer.register_forward_hook(self.get_target_input)
        self.target_handle = input_handle

    def add_handles(self, model, forward_handle, backward_handle):
        """
        Add handles to all non-container layers in the model.
        Recursively for non-container layers
        """
        handles_list = []
        for child in model.children():
            if 'nn.modules.container' in str(type(child)):
                handles_list.extend(self.add_handles(child, forward_handle, backward_handle))
            else:
                handles_list.append(child.register_forward_hook(forward_handle))
                handles_list.append(child.register_backward_hook(backward_handle))
        return handles_list

    def remove_attributes(self, model):
        """
        Removes the x and y attributes which were added by the forward handles
        Recursively searches for non-container layers
        """
        for child in model.children():
            if 'nn.modules.container' in str(type(child)):
                self.remove_attributes(child)
            else:
                try:
                    del child.x
                except AttributeError:
                    pass
                try:
                    del child.y
                except AttributeError:
                    pass

    @staticmethod
    def get_target_input(module, input, output):
        """Saves the tensor - attached to its graph.
        Used if we want to explain the interim outputs of a model
        """
        try:
            del module.target_input
        except AttributeError:
            pass
        setattr(module, 'target_input', input)

    @staticmethod
    def add_interim_values(module, input, output):
        """If necessary, saves interim tensors detached from the graph.
        Used to calculate multipliers
        """
        try:
            del module.x
        except AttributeError:
            pass
        try:
            del module.y
        except AttributeError:
            pass
        module_type = module.__class__.__name__
        if module_type in op_handler:
            func_name = op_handler[module_type].__name__
            # First, check for cases where we don't need to save the x and y tensors
            if func_name == 'passthrough':
                pass
            else:
                # check only the 0th input varies
                for i in range(len(input)):
                    if i != 0 and type(output) is tuple:
                        assert input[i] == output[i], "Only the 0th input may vary!"
                # if a new method is added, it must be added here too. This ensures tensors
                # are only saved if necessary
                if func_name in ['maxpool', 'nonlinear_1d']:
                    # only save tensors if necessary
                    if type(input) is tuple:
                        setattr(module, 'x', torch.nn.Parameter(input[0].detach()))
                    else:
                        setattr(module, 'x', torch.nn.Parameter(input.detach()))
                    if type(output) is tuple:
                        setattr(module, 'y', torch.nn.Parameter(output[0].detach()))
                    else:
                        setattr(module, 'y', torch.nn.Parameter(output.detach()))

    @staticmethod
    def deeplift_grad(module, grad_input, grad_output):
        # first, get the module type
        module_type = module.__class__.__name__
        # first, check the module is supported
        if module_type in op_handler:
            if op_handler[module_type].__name__ not in ['passthrough', 'linear_1d']:
                return op_handler[module_type](module, grad_input, grad_output)
        else:
            print('Warning: unrecognized nn.Module: {}'.format(module_type))
            return grad_input

    def gradient(self, idx, inputs):
        self.model.zero_grad()
        X = [x.requires_grad_() for x in inputs]
        outputs = self.model(*X)
        selected = [val for val in outputs[:, idx]]
        if self.interim:
            interim_inputs = self.layer.target_input
            grads = [torch.autograd.grad(selected, input)[0].cpu().numpy() for input in interim_inputs]
            del self.layer.target_input
            return grads, [i.detach().cpu().numpy() for i in interim_inputs]
        else:
            grads = [torch.autograd.grad(selected, x)[0].cpu().numpy() for x in X]
            return grads

    def shap_values(self, X, ranked_outputs=None, output_rank_order="max"):

        # X ~ self.model_input
        # X_data ~ self.data

        # check if we have multiple inputs
        if not self.multi_input:
            assert type(X) != list, "Expected a single tensor model input!"
            X = [X]
        else:
            assert type(X) == list, "Expected a list of model inputs!"

        if ranked_outputs is not None and self.multi_output:
            with torch.no_grad():
                model_output_values = self.model(*X)
            # rank and determine the model outputs that we will explain
            if output_rank_order == "max":
                _, model_output_ranks = torch.sort(model_output_values, descending=True)
            elif output_rank_order == "min":
                _, model_output_ranks = torch.sort(model_output_values, descending=False)
            elif output_rank_order == "max_abs":
                _, model_output_ranks = torch.sort(torch.abs(model_output_values), descending=True)
            else:
                assert False, "output_rank_order must be max, min, or max_abs!"
            model_output_ranks = model_output_ranks[:, :ranked_outputs]
        else:
            model_output_ranks = (torch.ones((X[0].shape[0], self.num_outputs)).int() *
                                  torch.arange(0, self.num_outputs).int())

        # add the gradient handles
        handles = self.add_handles(self.model, self.add_interim_values, self.deeplift_grad)
        if self.interim:
            self.add_target_handle(self.layer)

        # compute the attributions
        output_phis = []
        for i in range(model_output_ranks.shape[1]):
            phis = []
            if self.interim:
                for k in range(len(self.interim_inputs_shape)):
                    phis.append(np.zeros((X[0].shape[0], ) + self.interim_inputs_shape[k][1: ]))
            else:
                for k in range(len(X)):
                    phis.append(np.zeros(X[k].shape))
            for j in range(X[0].shape[0]):
                # tile the inputs to line up with the background data samples
                tiled_X = [X[l][j:j + 1].repeat(
                                   (self.data[l].shape[0],) + tuple([1 for k in range(len(X[l].shape) - 1)])) for l
                           in range(len(X))]
                print("-"*80)
                print("tiled_X len: ", len(tiled_X))
                print("tiled_X[0].shape: ", tiled_X[0].shape)
                print("-"*80)                           
                joint_x = [torch.cat((tiled_X[l], self.data[l]), dim=0) for l in range(len(X))]
                # run attribution computation graph
                feature_ind = model_output_ranks[j, i]
                sample_phis = self.gradient(feature_ind, joint_x)
                # assign the attributions to the right part of the output arrays
                if self.interim:
                    sample_phis, output = sample_phis
                    x, data = [], []
                    for i in range(len(output)):
                        x_temp, data_temp = np.split(output[i], 2)
                        x.append(x_temp)
                        data.append(data_temp)
                    for l in range(len(self.interim_inputs_shape)):
                        phis[l][j] = (sample_phis[l][self.data[l].shape[0]:]* (x[l] - data[l])).mean(0)
                else:
                    for l in range(len(X)):
                        phis[l][j] = (torch.from_numpy(sample_phis[l][self.data[l].shape[0]:]) * (X[l][j: j + 1] - self.data[l])).mean(0)
            output_phis.append(phis[0] if not self.multi_input else phis)
        # cleanup; remove all gradient handles
        for handle in handles:
            handle.remove()
        self.remove_attributes(self.model)
        if self.interim:
            self.target_handle.remove()

        if not self.multi_output:
            return output_phis[0]
        elif ranked_outputs is not None:
            return output_phis, model_output_ranks
        else:
            return output_phis

class PytorchBertDeepExplainer(Explainer):

    def __init__(self, model, mask, seg, tk_idx, device, vocabsize, id2onehot):
        # try and import pytorch
        global torch
        if torch is None:
            import torch
            if LooseVersion(torch.__version__) < LooseVersion("0.4"):
                warnings.warn("Your PyTorch version is older than 0.4 and not supported.")
        # self.device = torch.device("cuda:2")
        self.device = device
        # check if we have multiple inputs
        self.multi_input = False
        # if type(data) == list:
        #     self.multi_input = True
        # if type(data) != list:
        #     data = [data]
        self.id2onehot = id2onehot
        data = id2onehot(tk_idx, vocabsize)
        self.vocabsize = vocabsize
        self.data = [data]
        
        self.mask = mask
        self.seg = seg
        self.tk_idx = tk_idx
        self.layer = None
        self.input_handle = None
        self.interim = False
        self.interim_inputs_shape = None
        self.expected_value = None  # to keep the DeepExplainer base happy
        if type(model) == tuple:
            self.interim = True
            model, layer = model
            model = model.eval()
            self.layer = layer
            self.add_target_handle(self.layer)

            # if we are taking an interim layer, the 'data' is going to be the input
            # of the interim layer; we will capture this using a forward hook
            with torch.no_grad():
                _ = model(*data)
                interim_inputs = self.layer.target_input
                if type(interim_inputs) is tuple:
                    # this should always be true, but just to be safe
                    self.interim_inputs_shape = [i.shape for i in interim_inputs]
                else:
                    self.interim_inputs_shape = [interim_inputs.shape]
            self.target_handle.remove()
            del self.layer.target_input
        self.model = model.eval()

        self.multi_output = False
        self.num_outputs = 1
        with torch.no_grad():
            batch_size = 4
            max_batch_size = len(data) // batch_size
            print("feed forward start: ")
            print("-"*80)
            logits = []
            for i in tqdm(range(max_batch_size)):
                input_ids = tk_idx[i*batch_size: (i+1)*batch_size].to(self.device)
                onehot_input = data[i*batch_size: (i+1)*batch_size].to(self.device)
                input_mask = mask[i*batch_size: (i+1)*batch_size].to(self.device)
                segment_ids = seg[i*batch_size: (i+1)*batch_size].to(self.device)
                # labels = data[i*batch_size: (i+1)*batch_size][3].to(device)

                output = model(input_ids, segment_ids, input_mask, labels=None, one_hot=onehot_input)
                logits.append(output)
                # clean up to free some mem
                del input_ids, onehot_input, input_mask, segment_ids
                torch.cuda.empty_cache() 

            outputs = torch.cat(logits, dim=0)
            # outputs = model(*data)
            if outputs.shape[1] > 1:
                self.multi_output = True
                self.num_outputs = outputs.shape[1]

    def add_target_handle(self, layer):
        input_handle = layer.register_forward_hook(self.get_target_input)
        self.target_handle = input_handle

    def add_handles(self, model, forward_handle, backward_handle):
        """
        Add handles to all non-container layers in the model.
        Recursively for non-container layers
        """
        handles_list = []
        for child in model.children():
            if 'nn.modules.container' in str(type(child)):
                handles_list.extend(self.add_handles(child, forward_handle, backward_handle))
            else:
                handles_list.append(child.register_forward_hook(forward_handle))
                handles_list.append(child.register_backward_hook(backward_handle))
        return handles_list

    def remove_attributes(self, model):
        """
        Removes the x and y attributes which were added by the forward handles
        Recursively searches for non-container layers
        """
        for child in model.children():
            if 'nn.modules.container' in str(type(child)):
                self.remove_attributes(child)
            else:
                try:
                    del child.x
                except AttributeError:
                    pass
                try:
                    del child.y
                except AttributeError:
                    pass

    @staticmethod
    def get_target_input(module, input, output):
        """Saves the tensor - attached to its graph.
        Used if we want to explain the interim outputs of a model
        """
        try:
            del module.target_input
        except AttributeError:
            pass
        setattr(module, 'target_input', input)

    @staticmethod
    def add_interim_values(module, input, output):
        """If necessary, saves interim tensors detached from the graph.
        Used to calculate multipliers
        """
        try:
            del module.x
        except AttributeError:
            pass
        try:
            del module.y
        except AttributeError:
            pass
        module_type = module.__class__.__name__
        if module_type in op_handler:
            func_name = op_handler[module_type].__name__
            # First, check for cases where we don't need to save the x and y tensors
            if func_name == 'passthrough':
                pass
            else:
                # check only the 0th input varies
                for i in range(len(input)):
                    if i != 0 and type(output) is tuple:
                        assert input[i] == output[i], "Only the 0th input may vary!"
                # if a new method is added, it must be added here too. This ensures tensors
                # are only saved if necessary
                if func_name in ['maxpool', 'nonlinear_1d']:
                    # only save tensors if necessary
                    if type(input) is tuple:
                        setattr(module, 'x', torch.nn.Parameter(input[0].detach()))
                    else:
                        setattr(module, 'x', torch.nn.Parameter(input.detach()))
                    if type(output) is tuple:
                        setattr(module, 'y', torch.nn.Parameter(output[0].detach()))
                    else:
                        setattr(module, 'y', torch.nn.Parameter(output.detach()))

    @staticmethod
    def deeplift_grad(module, grad_input, grad_output):
        # first, get the module type
        module_type = module.__class__.__name__
        # first, check the module is supported
        if module_type in op_handler:
            if op_handler[module_type].__name__ not in ['passthrough', 'linear_1d']:
                return op_handler[module_type](module, grad_input, grad_output)
        else:
            # print('Warning: unrecognized nn.Module: {}'.format(module_type))
            return grad_input

    def gradient(self, idx, mask=None, seg=None,tk_idx=None):
        self.model.zero_grad()
        # X = [x.requires_grad_() for x in inputs]
        print("==========gradient===========")
        
        # outputs = self.model(*X)
        batch_size = 1
        max_batch_size = len(tk_idx[0]) // batch_size
        print("evaluation start: ")
        print("-"*80)
        #feature_ind, joint_x, joint_mask, joint_seg, joint_ids

        logits = []
        grads = []
        for i in tqdm(range(max_batch_size)):
            ids = tk_idx[0][i*batch_size: (i+1)*batch_size]
            onehots = self.id2onehot(ids, self.vocabsize)
            input_ids = ids.to(self.device)
            onehot_input = onehots.to(self.device).requires_grad_()
            input_mask = mask[0][i*batch_size: (i+1)*batch_size].to(self.device)
            segment_ids = seg[0][i*batch_size: (i+1)*batch_size].to(self.device)

            output = self.model(input_ids, segment_ids, input_mask, labels=None, one_hot=onehot_input)
            selected = [output[0, idx]]
            grads.append(torch.autograd.grad(selected, onehot_input)[0].cpu().numpy())

            # labels = data[i*batch_size: (i+1)*batch_size][3].to(device)
        grad_npmatrix = np.stack(grads)
        return [grad_npmatrix]

    def shap_values(self, X, eval_mask, seg, tk_idx, ranked_outputs=None, output_rank_order="max"):

        # X ~ self.model_input
        # X_data ~ self.data

        # check if we have multiple inputs
        if not self.multi_input:
            assert type(X) != list, "Expected a single tensor model input!"
            X = [X]
        else:
            assert type(X) == list, "Expected a list of model inputs!"

        if ranked_outputs is not None and self.multi_output:
            with torch.no_grad():
                model_output_values = self.model(*X)
            # rank and determine the model outputs that we will explain
            if output_rank_order == "max":
                _, model_output_ranks = torch.sort(model_output_values, descending=True)
            elif output_rank_order == "min":
                _, model_output_ranks = torch.sort(model_output_values, descending=False)
            elif output_rank_order == "max_abs":
                _, model_output_ranks = torch.sort(torch.abs(model_output_values), descending=True)
            else:
                assert False, "output_rank_order must be max, min, or max_abs!"
            model_output_ranks = model_output_ranks[:, :ranked_outputs]
        else:
            model_output_ranks = (torch.ones((X[0].shape[0], self.num_outputs)).int() *
                                  torch.arange(0, self.num_outputs).int())
            
        # add the gradient handles
        handles = self.add_handles(self.model, self.add_interim_values, self.deeplift_grad)
        if self.interim:
            self.add_target_handle(self.layer)

        # compute the attributions
        output_phis = []
        eval_mask = [eval_mask]
        eval_seg = [seg]
        ids = [tk_idx]
        
        for i in range(model_output_ranks.shape[1]):
            phis = []
            if self.interim:
                for k in range(len(self.interim_inputs_shape)):
                    phis.append(np.zeros((X[0].shape[0], ) + self.interim_inputs_shape[k][1: ]))
            else:
                for k in range(len(X)):
                    phis.append(np.zeros(X[k].shape))
            for j in range(X[0].shape[0]):
                # tile the inputs to line up with the background data samples
                
                # tiled_X = [X[l][j:j + 1].repeat(
                #                    (self.data[l].shape[0],) + tuple([1 for k in range(len(X[l].shape) - 1)])) for l
                #            in range(len(X))]
                
                # joint_x = [torch.cat((tiled_X[l], self.data[l]), dim=0) for l in range(len(X))]
                # run attribution computation graph
                feature_ind = model_output_ranks[j, i]
                #================mask===================
                
                
                tiled_eval_mask = [eval_mask[l][j:j + 1].repeat((self.data[l].shape[0],) + tuple([1 for k in range(len(eval_mask[l].shape) - 1)])) for l
                           in range(len(X))]
                
                joint_mask = [torch.cat((tiled_eval_mask[l], self.mask), dim=0) for l in range(len(X))]
                
                #=================end mask===============
                #=================seg===================
                
                tiled_eval_seg = [eval_seg[l][j:j + 1].repeat((self.data[l].shape[0],) + tuple([1 for k in range(len(eval_seg[l].shape) - 1)])) for l
                           in range(len(X))]
                
                joint_seg = [torch.cat((tiled_eval_seg[l], self.seg), dim=0) for l in range(len(X))]
                #=================end segs===============
                #==================ids===================
                
                tiled_eval_ids = [ids[l][j:j + 1].repeat((self.data[l].shape[0],) + tuple([1 for k in range(len(ids[l].shape) - 1)])) for l
                           in range(len(X))]
                
                joint_ids = [torch.cat((tiled_eval_ids[l], self.tk_idx), dim=0) for l in range(len(X))]
                #=================end ids===============
                # sample_phis = self.gradient(feature_ind, joint_x, joint_mask, joint_seg, joint_ids)
                sample_phis = self.gradient(feature_ind, joint_mask, joint_seg, joint_ids)
                # assign the attributions to the right part of the output arrays
                if self.interim:
                    sample_phis, output = sample_phis
                    x, data = [], []
                    for i in range(len(output)):
                        x_temp, data_temp = np.split(output[i], 2)
                        x.append(x_temp)
                        data.append(data_temp)
                    for l in range(len(self.interim_inputs_shape)):
                        phis[l][j] = (sample_phis[l][self.data[l].shape[0]:]* (x[l] - data[l])).mean(0)
                else:
                    for l in range(len(X)):
                        print("calc phis...")
                        test_phis = torch.from_numpy(sample_phis[l][self.data[l].shape[0]:])
                        x_diff = (X[l][j: j + 1] - self.data[l]).cpu()
                        temp_phis = []
                        for v in tqdm(range(100)):

                            temp_phis.append(test_phis[v] * x_diff[v])
                        del test_phis, x_diff
                        cat_phis = torch.cat(temp_phis, dim=0).mean(0)
                        # phis[l][j] = (torch.from_numpy(sample_phis[l][self.data[l].shape[0]:]) * (X[l][j: j + 1] - self.data[l]).cpu())
                        
                        phis[l][j] = cat_phis
                        
            output_phis.append(phis[0] if not self.multi_input else phis)
        # cleanup; remove all gradient handles

        for handle in handles:
            handle.remove()
        self.remove_attributes(self.model)
        if self.interim:
            self.target_handle.remove()

        if not self.multi_output:
            return output_phis[0]
        elif ranked_outputs is not None:
            return output_phis, model_output_ranks
        else:
            return output_phis

def passthrough(module, grad_input, grad_output):
    """No change made to gradients"""
    return None


def maxpool(module, grad_input, grad_output):
    pool_to_unpool = {
        'MaxPool1d': torch.nn.functional.max_unpool1d,
        'MaxPool2d': torch.nn.functional.max_unpool2d,
        'MaxPool3d': torch.nn.functional.max_unpool3d
    }
    pool_to_function = {
        'MaxPool1d': torch.nn.functional.max_pool1d,
        'MaxPool2d': torch.nn.functional.max_pool2d,
        'MaxPool3d': torch.nn.functional.max_pool3d
    }
    delta_in = module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2):]
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    # we also need to check if the output is a tuple
    y, ref_output = torch.chunk(module.y, 2)
    cross_max = torch.max(y, ref_output)
    diffs = torch.cat([cross_max - ref_output, y - cross_max], 0)

    # all of this just to unpool the outputs
    with torch.no_grad():
        _, indices = pool_to_function[module.__class__.__name__](
            module.x, module.kernel_size, module.stride, module.padding,
            module.dilation, module.ceil_mode, True)
        xmax_pos, rmax_pos = torch.chunk(pool_to_unpool[module.__class__.__name__](
            grad_output[0] * diffs, indices, module.kernel_size, module.stride,
            module.padding, delta_in.shape), 2)
    grad_input = [None for _ in grad_input]
    grad_input[0] = torch.where(torch.abs(delta_in) < 1e-7, torch.zeros_like(delta_in),
                           (xmax_pos + rmax_pos) / delta_in).repeat(dup0)
    # delete the attributes
    del module.x
    del module.y
    return tuple(grad_input)


def linear_1d(module, grad_input, grad_output):
    """No change made to gradients."""
    return None


def nonlinear_1d(module, grad_input, grad_output):
    delta_out = module.y[: int(module.y.shape[0] / 2)] - module.y[int(module.y.shape[0] / 2):]

    delta_in = module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2):]
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    # handles numerical instabilities where delta_in is very small by
    # just taking the gradient in those cases
    grads = [None for _ in grad_input]
    grads[0] = torch.where(torch.abs(delta_in.repeat(dup0)) < 1e-6, grad_input[0],
                           grad_output[0] * (delta_out / delta_in).repeat(dup0))

    # delete the attributes
    del module.x
    del module.y
    return tuple(grads)


op_handler = {}

# passthrough ops, where we make no change to the gradient
op_handler['Dropout3d'] = passthrough
op_handler['Dropout2d'] = passthrough
op_handler['Dropout'] = passthrough
op_handler['AlphaDropout'] = passthrough

op_handler['Conv2d'] = linear_1d
op_handler['Linear'] = linear_1d
op_handler['AvgPool1d'] = linear_1d
op_handler['AvgPool2d'] = linear_1d
op_handler['AvgPool3d'] = linear_1d
op_handler['BatchNorm1d'] = linear_1d
op_handler['BatchNorm2d'] = linear_1d
op_handler['BatchNorm3d'] = linear_1d

op_handler['ReLU'] = nonlinear_1d
op_handler['ELU'] = nonlinear_1d
op_handler['Sigmoid'] = nonlinear_1d
op_handler["Tanh"] = nonlinear_1d
op_handler["Softplus"] = nonlinear_1d
op_handler['Softmax'] = nonlinear_1d

op_handler['MaxPool1d'] = maxpool
op_handler['MaxPool2d'] = maxpool
op_handler['MaxPool3d'] = maxpool
