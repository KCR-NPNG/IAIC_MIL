import numpy as np
import torch
import cv2
import torch.nn as nn
from math import ceil
from scipy.stats import rankdata
from sobolAnakysis.sampler import ScipySobolSequence
from sobolAnakysis.estimator import inpainting, JansenEstimator



def resize(image, shape):
    return cv2.resize(image, shape, interpolation=cv2.INTER_CUBIC)

class SobolAttributionMethod:
    """
    Sobol' Attribution Method.

    Once the explainer is initialized, you can call it with an array of inputs and labels (int) 
    to get the STi.

    Parameters
    ----------
    grid_size: int, optional
        Cut the image in a grid of grid_size*grid_size to estimate an indice per cell.
    nb_design: int, optional
        Must be a power of two. Number of design, the number of forward will be nb_design(grid_size**2+2).
    sampler : Sampler, optional
        Sampler used to generate the (quasi-)monte carlo samples.
    estimator: Estimator, optional
        Estimator used to compute the total order sobol' indices.
    perturbation_function: function, optional
        Function to call to apply the perturbation on the input.
    batch_size: int, optional,
        Batch size to use for the forwards.
    """

    def __init__(
        self,
        model,
        grid_size=8,
        nb_design=8,
        sampler=ScipySobolSequence(),
        estimator=JansenEstimator(),
        perturbation_function=inpainting,
        mask_batch=128,
        select_num=5
    ):

        assert (nb_design & (nb_design-1) == 0) and nb_design != 0,\
            "The number of design must be a power of two."

        self.model = model

        self.grid_size = grid_size
        self.nb_design = nb_design
        self.select_num = select_num
        self.perturbation_function = perturbation_function

        self.sampler = sampler
        self.estimator = estimator
        self.loss = nn.CrossEntropyLoss().to('cuda')
        self.mask_batch = mask_batch
        masks = sampler(grid_size, nb_design)
        self.masks = torch.Tensor(masks).cuda()

    def __call__(self, inputs, labels):
        """
        Explain a particular prediction

        Parameters
        ----------
        inputs: ndarray or tf.Tensor [Nb_samples, Width, Height, Channels]
            Images to explain.
        labels: list of int,
            Label of the class to explain.
        """
       
        feat_dim = inputs.shape[1:] 
        perturbator = self.perturbation_function(inputs)
        y = np.zeros((len(self.masks)))
        nb_batch = ceil(len(self.masks) / self.mask_batch)

        for batch_index in range(nb_batch):
            start_index = batch_index * self.mask_batch
            end_index = min(len(self.masks), (batch_index+1)*self.mask_batch)
            batch_masks = self.masks[start_index:end_index]

            # 进行敏感性分析
            batch_y = SobolAttributionMethod._batch_forward(self.model, labels, batch_masks,
                                                                perturbator, self.select_num)
            batch_y = np.array(batch_y)
            y[start_index:end_index] = batch_y
        sti = self.estimator(self.masks, y, self.nb_design) # (2112,1,8,8)(2112,)(32)
        return sti
    
    @staticmethod
    def _batch_forward(model, labels, masks, perturbator, select_num):

        # upsampled_masks = masks.unsqueeze(2).expand(-1,-1, feat_dim).clone() # 将mask的列复制feat_dim次得到maskbatch*batchsize*featuredim维度
        
        perturbated_inputs = perturbator(masks) # 返回原始输入和扰动后的输入
        batch_size, num_patches, feature_dim = perturbated_inputs.shape
        perturbated_inputs_reshaped = perturbated_inputs.reshape(-1, feature_dim)
        # for idx in range(perturbated_inputs.shape[0]):
        output,_,_ = model(perturbated_inputs_reshaped) #
        output = output.reshape(batch_size, num_patches, -1) 
        selected_output = output[:, :, labels.item()]  # (batch_size, num_patches)
        k = min(selected_output.shape[1], select_num)
        top_k_values, _ = torch.topk(selected_output, k, dim=1)  # (batch_size, 5)
        top_k_mean = top_k_values.mean(dim=1)  # (batch_size,)
        return top_k_mean.cpu().detach().numpy()
        # top_k_output, _ = torch.topk(output[:,labels.item()], 5)
        # top_k_mean = top_k_output.mean().cpu().detach().numpy()
        # outputs.append(top_k_mean)
        # return outputs 

def sobol_analysis(tattFeats, label, classifier, params):

    batch_size, feature_dim = tattFeats.shape
    # print(tattFeats.shape)
    explainer = SobolAttributionMethod(classifier, grid_size=batch_size, nb_design=params.nb_design, mask_batch=params.mask_batch, select_num=params.select_num )
    sti = explainer(tattFeats, label)
    return sti