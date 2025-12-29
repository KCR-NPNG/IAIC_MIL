from abc import ABC, abstractmethod
import numpy as np
import torch

def _baseline_ponderation(x, masks, x0):
    
    perturbated_inputs = x[None, :, :] * masks[:,:,None] 
    return perturbated_inputs

def inpainting(input):
    """
    Tensorflow inpainting perturbation function.

    X_perturbed = X * M

    Parameters
    ----------
    input: tf.Tensor
        Image to perform perturbation on.

    Returns
    -------
    f: callable
        Inpainting perturbation function.
    """
    # if method == "zero":
    #     x0 = torch.zeros_like(input).to(input.device)  # 全零填充
    # elif method == "mean":
    #     x0 = torch.mean(input, dim=0, keepdim=True).to(input.device)  # 均值填充
    # elif method == "noise":
    #     noise_level = 0.1
    #     x0 = input + noise_level * torch.randn_like(input)  # 高斯噪声填充
    # else:
    #     raise ValueError("method should be 'zero', 'mean' or 'noise'.")
    
    x0 = np.zeros(input.shape)
    x0 = torch.Tensor(x0).to(input.device)

    def f(masks):
        return _baseline_ponderation(input, masks, x0)
    return f



class SobolEstimator(ABC):
    """
    Base class for Sobol' total order estimators.
    """

    def _masks_dim(self, masks):
        """
        Deduce the number of dimensions using the sampling masks.

        Parameters
        ----------
        masks: ndarray
          Low resolution masks (before upsampling) used, one for each output.

        Returns
        -------
        nb_dim: int
          The number of dimensions under study according to the masks.
        """
        nb_dim = np.prod(masks.shape[1:])
        return nb_dim

    def _split_abc(self, outputs, nb_design, nb_dim):
        """
        Split the outputs values into the 3 sampling matrices A, B and C.

        Parameters
        ----------
        outputs: ndarray
          Model outputs for each sample point of matrices A, B and C (in order).
        nb_design: int
          Number of points for matrices A (the same as B).
        nb_dim: int
          Number of dimensions to estimate.

        Returns
        -------
        a: ndarray
          The results for the sample points in matrix A.
        b: ndarray
          The results for the sample points in matrix A.
        c: ndarray
          The results for the sample points in matrix C.
        """
        a = outputs[:nb_design]
        b = outputs[nb_design:nb_design*2]
        c = np.array([outputs[nb_design*2 + nb_design*i:nb_design*2 + nb_design*(i+1)]
                      for i in range(nb_dim)])
        return a, b, c

    def _post_process(self, stis, masks):
        """
        Post processing ops on the indices before sending them back. Makes sure the data
        format and shape is correct. 

        Parameters
        ----------
        stis: ndarray
          Total order Sobol' indices, one for each dimensions.
        masks: ndarray
            Low resolution masks (before upsampling) used, one for each output.

        Returns
        -------
        stis: ndarray
          Total order Sobol' indices after post processing.
        """
        stis = np.array(stis, np.float32)
        # return stis.reshape(masks.shape[1:])
        return stis

    @abstractmethod
    def __call__(self, masks, outputs, nb_design):
        """
        Compute the Sobol' total order indices according to the Jansen algorithm.

        Ref. Jansen, M., Analysis of variance designs for model output (1999)
        https://www.sciencedirect.com/science/article/abs/pii/S0010465598001544

        Parameters
        ----------
        masks: ndarray
          Low resolution masks (before upsampling) used, one for each output.
        outputs: ndarray
          Model outputs associated to each masks. One for each sample point of 
          matrices A, B and C (in order).
        nb_design: int
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti: ndarray
          Total order Sobol' indices, one for each dimensions.
        """
        raise NotImplementedError()


class JansenEstimator(SobolEstimator):
    """
    Jansen estimator for total order Sobol' indices.

    Ref. Jansen, M., Analysis of variance designs for model output (1999)
    https://www.sciencedirect.com/science/article/abs/pii/S0010465598001544
    """

    def __call__(self, masks, outputs, nb_design):
        """
        Compute the Sobol' total order indices according to the Jansen algorithm.

        Parameters
        ----------
        masks: ndarray
          Low resolution masks (before upsampling) used, one for each output.
        outputs: ndarray
          Model outputs associated to each masks. One for each sample point of 
          matrices A, B and C (in order).
        nb_design: int
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti: ndarray
          Total order Sobol' indices, one for each dimensions.
        """
        nb_dim = self._masks_dim(masks) # 507
        a, b, c = self._split_abc(outputs, nb_design, nb_dim)

        f0 = np.mean(a)
        epsilon = 1e-8
        var = np.sum([(v - f0)**2 for v in a]) / (len(a) - 1)
        var = max(var, epsilon)
        first_order = [
            np.sum((a - c[i])**2.0) / (2 * nb_design * var)
            for i in range(nb_dim)
        ]
        # 总效应 Sobol 敏感性指数（STi） - 使用 Jansen 估计器
        # STi = (1 / (2N)) * sum((b - c)^2) / Var(f)
        # total_order = [
        #     np.sum((b - c[i])**2.0) / (2 * nb_design * var)
        #     for i in range(nb_dim)
        # ]

        # alpha = 0.5  # 可调节，建议从0.5或0.7开始测试
        # combined_scores = [
        #     alpha * fi + (1 - alpha) * ti
        #     for fi, ti in zip(first_order, total_order)
        # ]

        return self._post_process(first_order, masks)
