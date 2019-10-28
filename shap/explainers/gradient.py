import numpy as np
import warnings
from .explainer import Explainer
from distutils.version import LooseVersion
keras = None
tf = None
torch = None


class GradientExplainer(Explainer):
    """ Explains a model using expected gradients (an extension of integrated gradients).

    Expected gradients an extension of the integrated gradients method (Sundararajan et al. 2017), a
    feature attribution method designed for differentiable models based on an extension of Shapley
    values to infinite player games (Aumann-Shapley values). Integrated gradients values are a bit
    different from SHAP values, and require a single reference value to integrate from. As an adaptation
    to make them approximate SHAP values, expected gradients reformulates the integral as an expectation
    and combines that expectation with sampling reference values from the background dataset. This leads
    to a single combined expectation of gradients that converges to attributions that sum to the
    difference between the expected model output and the current output.
    """

    def __init__(self, model, data, session=None, batch_size=50, local_smoothing=0):
        """ An explainer object for a differentiable model using a given background dataset.

        Note that the complexity of the method scales linearly with the number of background data
        samples. Passing the entire training dataset as `data` will give very accurate expected
        values, but be unreasonably expensive. The variance of the expectation estimates scale by
        roughly 1/sqrt(N) for N background data samples. So 100 samples will give a good estimate,
        and 1000 samples a very good estimate of the expected values.

        Parameters
        ----------
        model : if framework == 'tensorflow', (input : [tf.Operation], output : tf.Operation)
             A pair of TensorFlow operations (or a list and an op) that specifies the input and
            output of the model to be explained. Note that SHAP values are specific to a single
            output value, so the output tf.Operation should be a single dimensional output (,1).

            if framework == 'pytorch', an nn.Module object (model), or a tuple (model, layer),
                where both are nn.Module objects
            The model is an nn.Module object which takes as input a tensor (or list of tensors) of
            shape data, and returns a single dimensional output.
            If the input is a tuple, the returned shap values will be for the input of the
            layer argument. layer must be a layer in the model, i.e. model.conv2

        data :
            if framework == 'tensorflow': [numpy.array] or [pandas.DataFrame]
            if framework == 'pytorch': [torch.tensor]
            The background dataset to use for integrating out features. GradientExplainer integrates
            over these samples. The data passed here must match the input operations given in the
            first argument.
        """

        # first, we need to find the framework
        if type(model) is tuple:
            a, b = model
            try:
                a.named_parameters()
                framework = 'pytorch'
            except:
                framework = 'tensorflow'
        else:
            try:
                model.named_parameters()
                framework = 'pytorch'
            except:
                framework = 'tensorflow'


        if framework == 'tensorflow':
            self.explainer = _TFGradientExplainer(model, data, session, batch_size, local_smoothing)
        elif framework == 'pytorch':
            self.explainer = _PyTorchGradientExplainer(model, data, batch_size, local_smoothing)

    def shap_values(self, X, nsamples=200, ranked_outputs=None, output_rank_order="max", rseed=None):
        """ Return the values for the model applied to X.

        Parameters
        ----------
        X : list,
            if framework == 'tensorflow': numpy.array, or pandas.DataFrame
            if framework == 'pytorch': torch.tensor
            A tensor (or list of tensors) of samples (where X.shape[0] == # samples) on which to
            explain the model's output.

        ranked_outputs : None or int
            If ranked_outputs is None then we explain all the outputs in a multi-output model. If
            ranked_outputs is a positive integer then we only explain that many of the top model
            outputs (where "top" is determined by output_rank_order). Note that this causes a pair
            of values to be returned (shap_values, indexes), where phi is a list of numpy arrays for each of
            the output ranks, and indexes is a matrix that tells for each sample which output indexes
            were choses as "top".

        output_rank_order : "max", "min", "max_abs", or "custom"
            How to order the model outputs when using ranked_outputs, either by maximum, minimum, or
            maximum absolute value. If "custom" Then "ranked_outputs" contains a list of output nodes.

        rseed : None or int
            Seeding the randomness in shap value computation  (background example choice, 
            interpolation between current and background example, smoothing).

        Returns
        -------
        For a models with a single output this returns a tensor of SHAP values with the same shape
        as X. For a model with multiple outputs this returns a list of SHAP value tensors, each of
        which are the same shape as X. If ranked_outputs is None then this list of tensors matches
        the number of model outputs. If ranked_outputs is a positive integer a pair is returned
        (shap_values, indexes), where shap_values is a list of tensors with a length of
        ranked_outputs, and indexes is a matrix that tells for each sample which output indexes
        were chosen as "top".
        """
        return self.explainer.shap_values(X, nsamples, ranked_outputs, output_rank_order, rseed)


class _TFGradientExplainer(Explainer):

    def __init__(self, model, data, session=None, batch_size=50, local_smoothing=0):

        # try and import keras and tensorflow
        global tf, keras
        if tf is None:
            import tensorflow as tf
            if LooseVersion(tf.__version__) < LooseVersion("1.4.0"):
                warnings.warn("Your TensorFlow version is older than 1.4.0 and not supported.")
        if keras is None:
            try:
                import keras
                if LooseVersion(keras.__version__) < LooseVersion("2.1.0"):
                    warnings.warn("Your Keras version is older than 2.1.0 and not supported.")
            except:
                pass

        # determine the model inputs and outputs
        if str(type(model)).endswith("keras.engine.sequential.Sequential'>"):
            self.model_inputs = model.inputs
            self.model_output = model.layers[-1].output
        elif str(type(model)).endswith("keras.models.Sequential'>"):
            self.model_inputs = model.inputs
            self.model_output = model.layers[-1].output
        elif str(type(model)).endswith("keras.engine.training.Model'>"):
            self.model_inputs = model.inputs
            self.model_output = model.layers[-1].output
        elif str(type(model)).endswith("tuple'>"):
            self.model_inputs = model[0]
            self.model_output = model[1]
        else:
            assert False, str(type(model)) + " is not currently a supported model type!"
        assert type(self.model_output) != list, "The model output to be explained must be a single tensor!"
        assert len(self.model_output.shape) < 3, "The model output must be a vector or a single value!"
        self.multi_output = True
        if len(self.model_output.shape) == 1:
            self.multi_output = False

        # check if we have multiple inputs
        self.multi_input = True
        if type(self.model_inputs) != list:
            self.multi_input = False
            self.model_inputs = [self.model_inputs]
        if type(data) != list:
            data = [data]

        self.data = data
        self._num_vinputs = {}
        self.batch_size = batch_size
        self.local_smoothing = local_smoothing

        # if we are not given a session find a default session
        if session is None:
            # if keras is installed and already has a session then use it
            if keras is not None and keras.backend.tensorflow_backend._SESSION is not None:
                session = keras.backend.get_session()
            else:
                session = tf.keras.backend.get_session()
        self.session = tf.get_default_session() if session is None else session

        # see if there is a keras operation we need to save
        self.keras_phase_placeholder = None
        for op in self.session.graph.get_operations():
            if 'keras_learning_phase' in op.name:
                self.keras_phase_placeholder = op.outputs[0]

        # save the expected output of the model
        #self.expected_value = self.run(self.model_output, self.model_inputs, self.data).mean(0)

        if not self.multi_output:
            self.gradients = [None]
        else:
            self.gradients = [None for i in range(self.model_output.shape[1])]

    def gradient(self, i):
        if self.gradients[i] is None:
            out = self.model_output[:,i] if self.multi_output else self.model_output
            self.gradients[i] = tf.gradients(out, self.model_inputs)
        return self.gradients[i]

    def shap_values(self, X, nsamples=200, ranked_outputs=None, output_rank_order="max", rseed=None):

        # check if we have multiple inputs
        if not self.multi_input:
            assert type(X) != list, "Expected a single tensor model input!"
            X = [X]
        else:
            assert type(X) == list, "Expected a list of model inputs!"
        assert len(self.model_inputs) == len(X), "Number of model inputs does not match the number given!"

        # rank and determine the model outputs that we will explain
        model_output_values = self.run(self.model_output, self.model_inputs, X)
        if ranked_outputs is not None and self.multi_output:
            if output_rank_order == "max":
                model_output_ranks = np.argsort(-model_output_values)
            elif output_rank_order == "min":
                model_output_ranks = np.argsort(model_output_values)
            elif output_rank_order == "max_abs":
                model_output_ranks = np.argsort(np.abs(model_output_values))
            elif output_rank_order == "custom":
                model_output_ranks = ranked_outputs
            else:
                assert False, "output_rank_order must be max, min, max_abs or custom!"

            if output_rank_order in ["max", "min", "max_abs"]:
                model_output_ranks = model_output_ranks[:,:ranked_outputs]
        else:
            model_output_ranks = np.tile(np.arange(len(self.gradients)), (X[0].shape[0], 1))

        # compute the attributions
        output_phis = []
        samples_input = [np.zeros((nsamples,) + X[l].shape[1:]) for l in range(len(X))]
        samples_delta = [np.zeros((nsamples,) + X[l].shape[1:]) for l in range(len(X))]
        # use random seed if no argument given
        if rseed is None:
            rseed = np.random.randint(0, 1e6)

        for i in range(model_output_ranks.shape[1]):
            np.random.seed(rseed) # so we get the same noise patterns for each output class
            phis = []
            phi_vars = []
            for k in range(len(X)):
                phis.append(np.zeros(X[k].shape))
                phi_vars.append(np.zeros(X[k].shape))
            for j in range(X[0].shape[0]):

                # fill in the samples arrays
                for k in range(nsamples):
                    rind = np.random.choice(self.data[0].shape[0])
                    t = np.random.uniform()
                    for l in range(len(X)):
                        if self.local_smoothing > 0:
                            x = X[l][j] + np.random.randn(*X[l][j].shape) * self.local_smoothing
                        else:
                            x = X[l][j]
                        samples_input[l][k] = t * x + (1 - t) * self.data[l][rind]
                        samples_delta[l][k] = x - self.data[l][rind]

                # compute the gradients at all the sample points
                find = model_output_ranks[j,i]
                grads = []
                for b in range(0, nsamples, self.batch_size):
                    batch = [samples_input[l][b:min(b+self.batch_size,nsamples)] for l in range(len(X))]
                    grads.append(self.run(self.gradient(find), self.model_inputs, batch))
                grad = [np.concatenate([g[l] for g in grads], 0) for l in range(len(X))]

                # assign the attributions to the right part of the output arrays
                for l in range(len(X)):
                    samples = grad[l] * samples_delta[l]
                    phis[l][j] = samples.mean(0)
                    phi_vars[l][j] = samples.var(0) / np.sqrt(samples.shape[0]) # estimate variance of means

                # TODO: this could be avoided by integrating between endpoints if no local smoothing is used
                # correct the sum of the values to equal the output of the model using a linear
                # regression model with priors of the coefficents equal to the estimated variances for each
                # value (note that 1e-6 is designed to increase the weight of the sample and so closely
                # match the correct sum)
                # if False and self.local_smoothing == 0: # disabled right now to make sure it doesn't mask problems
                #     phis_sum = np.sum([phis[l][j].sum() for l in range(len(X))])
                #     phi_vars_s = np.stack([phi_vars[l][j] for l in range(len(X))], 0).flatten()
                #     if self.multi_output:
                #         sum_error = model_output_values[j,find] - phis_sum - self.expected_value[find]
                #     else:
                #         sum_error = model_output_values[j] - phis_sum - self.expected_value

                #     # this is a ridge regression with one sample of all ones with sum_error as the label
                #     # and 1/v as the ridge penalties. This simlified (and stable) form comes from the
                #     # Sherman-Morrison formula
                #     v = (phi_vars_s / phi_vars_s.max()) * 1e6
                #     adj = sum_error * (v - (v * v.sum()) / (1 + v.sum()))

                #     # add the adjustment to the output so the sum matches
                #     offset = 0
                #     for l in range(len(X)):
                #         s = np.prod(phis[l][j].shape)
                #         phis[l][j] += adj[offset:offset+s].reshape(phis[l][j].shape)
                #         offset += s

            output_phis.append(phis[0] if not self.multi_input else phis)
        if not self.multi_output:
            return output_phis[0]
        elif ranked_outputs is not None:
            return output_phis, model_output_ranks
        else:
            return output_phis

    def run(self, out, model_inputs, X):
        feed_dict = dict(zip(model_inputs, X))
        if self.keras_phase_placeholder is not None:
            feed_dict[self.keras_phase_placeholder] = 0
        return self.session.run(out, feed_dict)


class _PyTorchGradientExplainer(Explainer):

    def __init__(self, model, data, batch_size=50, local_smoothing=0):

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

        # for consistency, the method signature calls for data as the model input.
        # However, within this class, self.model_inputs is the input (i.e. the data passed by the user)
        # and self.data is the background data for the layer we want to assign importances to. If this layer is
        # the input, then self.data = self.model_inputs
        self.model_inputs = data
        self.batch_size = batch_size
        self.local_smoothing = local_smoothing

        self.layer = None
        self.input_handle = None
        self.interim = False
        if type(model) == tuple:
            self.interim = True
            model, layer = model
            model = model.eval()
            self.add_handles(layer)
            self.layer = layer

            # now, if we are taking an interim layer, the 'data' is going to be the input
            # of the interim layer; we will capture this using a forward hook
            with torch.no_grad():
                _ = model(*data)
                interim_inputs = self.layer.target_input
                if type(interim_inputs) is tuple:
                    # this should always be true, but just to be safe
                    self.data = [i.clone().detach() for i in interim_inputs]
                else:
                    self.data = [interim_inputs.clone().detach()]
        else:
            self.data = data
        self.model = model.eval()

        multi_output = False
        outputs = self.model(*self.model_inputs)
        if outputs.shape[1] > 1:
            multi_output = True
        self.multi_output = multi_output

        if not self.multi_output:
            self.gradients = [None]
        else:
            self.gradients = [None for i in range(outputs.shape[1])]

    def gradient(self, idx, inputs):
        self.model.zero_grad()
        X = [x.requires_grad_() for x in inputs]
        outputs = self.model(*X)
        selected = [val for val in outputs[:, idx]]
        if self.input_handle is not None:
            interim_inputs = self.layer.target_input
            grads = [torch.autograd.grad(selected, input)[0].cpu().numpy() for input in interim_inputs]
            del self.layer.target_input
        else:
            grads = [torch.autograd.grad(selected, x)[0].cpu().numpy() for x in X]
        return grads

    @staticmethod
    def get_interim_input(self, input, output):
        try:
            del self.target_input
        except AttributeError:
            pass
        setattr(self, 'target_input', input)

    def add_handles(self, layer):
        input_handle = layer.register_forward_hook(self.get_interim_input)
        self.input_handle = input_handle

    def shap_values(self, X, nsamples=200, ranked_outputs=None, output_rank_order="max", rseed=None):

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
            model_output_ranks = (torch.ones((X[0].shape[0], len(self.gradients))).int() *
                                  torch.arange(0, len(self.gradients)).int())

        # if a cleanup happened, we need to add the handles back
        # this allows shap_values to be called multiple times, but the model to be
        # 'clean' at the end of each run for other uses
        if self.input_handle is None and self.interim is True:
            self.add_handles(self.layer)

        # compute the attributions
        X_batches = X[0].shape[0]
        output_phis = []
        # samples_input = input to the model
        # samples_delta = (x - x') for the input being explained - may be an interim input
        samples_input = [torch.zeros((nsamples,) + X[l].shape[1:], device=X[l].device) for l in range(len(X))]
        samples_delta = [np.zeros((nsamples, ) + self.data[l].shape[1:]) for l in range(len(self.data))]

        # use random seed if no argument given
        if rseed is None:
            rseed = np.random.randint(0, 1e6)

        for i in range(model_output_ranks.shape[1]):
            np.random.seed(rseed)  # so we get the same noise patterns for each output class
            phis = []
            phi_vars = []
            for k in range(len(self.data)):
                # for each of the inputs being explained - may be an interim input
                phis.append(np.zeros((X_batches,) + self.data[k].shape[1:]))
                phi_vars.append(np.zeros((X_batches, ) + self.data[k].shape[1:]))
            for j in range(X[0].shape[0]):
                # fill in the samples arrays
                for k in range(nsamples):
                    rind = np.random.choice(self.data[0].shape[0])
                    t = np.random.uniform()
                    for l in range(len(X)):
                        if self.local_smoothing > 0:
                            # local smoothing is added to the base input, unlike in the TF gradient explainer
                            x = X[l][j].clone().detach() + torch.empty(X[l][j].shape, device=X[l].device).normal_() \
                                * self.local_smoothing
                        else:
                            x = X[l][j].clone().detach()
                        samples_input[l][k] = (t * x + (1 - t) * (self.model_inputs[l][rind]).clone().detach()).\
                            clone().detach()
                        if self.input_handle is None:
                            samples_delta[l][k] = (x - (self.data[l][rind]).clone().detach()).cpu().numpy()

                    if self.interim is True:
                        with torch.no_grad():
                            _ = self.model(*[samples_input[l][k].unsqueeze(0) for l in range(len(X))])
                            interim_inputs = self.layer.target_input
                            del self.layer.target_input
                            if type(interim_inputs) is tuple:
                                if type(interim_inputs) is tuple:
                                    # this should always be true, but just to be safe
                                    for l in range(len(interim_inputs)):
                                        samples_delta[l][k] = interim_inputs[l].cpu().numpy()
                                else:
                                    samples_delta[0][k] = interim_inputs.cpu().numpy()

                # compute the gradients at all the sample points
                find = model_output_ranks[j, i]
                grads = []
                for b in range(0, nsamples, self.batch_size):
                    batch = [samples_input[l][b:min(b+self.batch_size,nsamples)].clone().detach() for l in range(len(X))]
                    grads.append(self.gradient(find, batch))
                grad = [np.concatenate([g[l] for g in grads], 0) for l in range(len(self.data))]
                # assign the attributions to the right part of the output arrays
                for l in range(len(self.data)):
                    samples = grad[l] * samples_delta[l]
                    phis[l][j] = samples.mean(0)
                    phi_vars[l][j] = samples.var(0) / np.sqrt(samples.shape[0]) # estimate variance of means

            output_phis.append(phis[0] if len(self.data) == 1 else phis)
        # cleanup: remove the handles, if they were added
        if self.input_handle is not None:
            self.input_handle.remove()
            self.input_handle = None
            # note: the target input attribute is deleted in the loop

        if not self.multi_output:
            return output_phis[0]
        elif ranked_outputs is not None:
            return output_phis, model_output_ranks
        else:
            return output_phis
