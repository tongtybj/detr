import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn
import math
import time
import importlib
import sys
import os
import cv2
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../external_tracker/external_module/pytracking'))

from pytracking.tracker.base import BaseTracker
from pytracking import dcf, fourier, TensorList, operation
from pytracking.features.featurebase import MultiFeatureBase
from pytracking.features.extractor import MultiResolutionExtractor
from pytracking.features.preprocessing import numpy_to_torch, sample_patch
from pytracking.utils.plotting import show_tensor
from pytracking.libs.optimization import GaussNewtonCG, ConjugateGradient, GradientDescentL2
from pytracking.features import augmentation
import ltr.data.bounding_box_utils as bbutils
from pytracking.tracker.atom.optim import ConvProblem, FactorizedConvProblem


# TrTr
import datasets.utils
from datasets.utils import crop_hwc, crop_image, siamfc_like_scale, get_exemplar_size, get_context_amount # TODO: move to utils
from external_tracker import build_external_tracker
from util.box_ops import box_cxcywh_to_xyxy
from util.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from models import build_model
from models.backbone import Backbone as Resnet



class MultiResolutionExtractor2(MultiResolutionExtractor):
    """Multi-resolution feature extractor with original padding model.
    args:
        features: List of features.
    """
    def __init__(self, features):
        super().__init__(features)

    def extract(self, image, pos, scales, image_sz, padding_value):
        """Extract features.
        args:
            im: Image.
            pos: Center position for extraction.
            scales: Image scales to extract features from.
            image_sz: Size to resize the image samples to before extraction.
            padding_value: the padding value for outside are in color 3 channel
        """
        if isinstance(scales, (int, float)):
            scales = [scales]


        # Get image patches
        # im = numpy_to_torch(image)
        # patch_iter, coord_iter = zip(*(sample_patch(im, pos, s*image_sz, image_sz, mode=self.patch_mode, max_scale_change=self.max_scale_change) for s in scales))
        # im_patches = torch.cat(list(patch_iter))

        im_patches = []
        for s in scales:
            assert image_sz[0] == image_sz[1] # TODO: change utils.crop_hwc
            sz = s * image_sz[0]
            bbox  = [pos[1]-sz/2, pos[0]-sz/2, pos[1]+sz/2, pos[0]+sz/2]
            crop_image = crop_hwc(image, bbox, image_sz[0], padding = padding_value)
            im_patches.append(numpy_to_torch(crop_image))
        im_patches = torch.cat(im_patches)

        # Compute features
        feature_map = TensorList([f.get_feature(im_patches) for f in self.features]).unroll()

        return feature_map

    def extract_transformed(self, image, pos, scale, image_sz, transforms, padding_value):
        """Extract features from a set of transformed image samples.
        args:
            im: Image.
            pos: Center position for extraction.
            scale: Image scale to extract features from.
            image_sz: Size to resize the image samples to before extraction.
            transforms: A set of image transforms to apply.
            padding_value: the padding value for outside are in color 3 channel
        """

        # Get image patche
        # im = numpy_to_torch(image)
        # im_patch, _ = sample_patch(im, pos, scale*image_sz, image_sz) # no padding

        assert image_sz[0] == image_sz[1] # TODO: change utils.crop_hwc
        sz = scale*image_sz[0]
        #print("extract transfromed sz: ", sz, ", original im: ", image.shape)
        bbox  = [pos[1]-sz/2, pos[0]-sz/2, pos[1]+sz/2, pos[0]+sz/2]
        crop_image = crop_hwc(image, bbox, image_sz[0], padding = padding_value)
        im_patch = numpy_to_torch(crop_image)

        # Apply transforms
        im_patches = torch.cat([T(im_patch) for T in transforms])

        # Compute features
        feature_map = TensorList([f.get_feature(im_patches) for f in self.features]).unroll()

        return feature_map


class ATOMResNet(MultiFeatureBase):
    """ResNet backbone wrapper to perform ATOM online updating 
    args:
        output_layers: List of layers to output.
        net_path: Relative or absolute net path (default should be fine).
    """
    def __init__(self, model_name = 'resnet18', output_layers=['layer3'], dilation = False, trtr_model = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_name = model_name
        self.output_layers = list(output_layers)
        self.use_gpu = True
        self.output_layers = output_layers
        self.net = Resnet(model_name, train_backbone = False, return_layers = output_layers , dilation = dilation)
        self.trtr_model = trtr_model
        self.net.cuda()
        self.net.eval()

    def initialize(self):
        if isinstance(self.pool_stride, int) and self.pool_stride == 1:
            self.pool_stride = [1] * len(self.output_layers)

        self.feature_layers = self.output_layers

        # for input
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

    def stride(self):
        stride = self.net.stride
        if self.model_name == 'resnet50': # TODO: special, decrease the depth and the width
            stride = stride * 2
        return TensorList([stride])

    def size(self, im_sz):
        if self.output_size is None:
            return TensorList([(im_sz + s - 1) // s for s in self.stride()])
        if isinstance(im_sz, torch.Tensor):
            return TensorList([(im_sz + s - 1) // s if sz is None else torch.Tensor([sz[0], sz[1]]) for sz, s in zip(self.output_size, self.stride())])


    def extract(self, im: torch.Tensor):
        im = im / 255
        im -= self.mean
        im /= self.std
        im = im.cuda()

        mask = [0, 0, im.shape[-2], im.shape[-1]]
        with torch.no_grad():
            xs = self.net(nested_tensor_from_tensor_list(im, [torch.as_tensor(mask).float()] * im.shape[0]))
            features = []
            for name, x in xs.items():
                if self.model_name == 'resnet50': # TODO: special, decrease the depth and the width
                    #features.append(F.adaptive_avg_pool2d(self.trtr_model.input_projs[int(name)](x.tensors), x.tensors.shape[-1]//2) )
                    features.append(F.adaptive_avg_pool2d(x.tensors, x.tensors.shape[-1]//2) )
                else:
                    features.append(x.tensors)

        return TensorList(features)


class Tracker():

    def __init__(self, model, postprocess, search_size, window_factor, score_threshold, window_steps, size_penalty_k, size_lpf):

        atom_param_module = importlib.import_module('pytracking.parameter.atom.default_vot')
        self.atom_params = atom_param_module.parameters()


        # TrTr model
        self.model = model
        self.model.eval()
        self.postprocess = postprocess

        self.search_size = search_size
        backbone_stride = model.backbone.stride
        self.heatmap_size = (search_size + backbone_stride - 1) // backbone_stride
        self.size_lpf = size_lpf
        self.size_penalty_k = size_penalty_k

        hanning = np.hanning(self.heatmap_size)
        self.window = torch.as_tensor(np.outer(hanning, hanning).flatten())
        self.window_factor = window_factor
        self.score_threshold = score_threshold
        self.window_steps = window_steps
        self.image_normalize = T.Compose([
            T.ToTensor(), # Converts a numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.first_frame = False

        # Get feature specific params
        self.atom_fparams = self.atom_params.features.get_fparams('feature_params')

        # overwrite the backbone for online updating (ATOM type)
        deep_feat = ATOMResNet(output_layers=['layer3'], fparams=self.atom_fparams, normalize_power=2)
        #deep_feat = ATOMResNet(model_name = 'resnet50', dilation = True, trtr_model = self.model, output_layers=['layer3'], fparams=self.atom_fparams, normalize_power=2)
        self.atom_params.features = MultiResolutionExtractor2([deep_feat])
        # todo: remember normalize_power (see MultiFeatureBase)?

        self.atom_params.train_skipping = 10 # TODO: tuning 10 - 20 (vot-toolkit)

    def init(self, image, bbox):

        # Get position and size
        # NOTE: if you use torch.Tensor, the default type is float32
        # https://stackoverflow.com/questions/48482787/pytorch-memory-model-torch-from-numpy-vs-torch-tensor
        # That is the reason why it has a slight worse performance than list operation whcih is  double64 type.
        # But right now it is OK
        self.target_pos = torch.Tensor([bbox[1] + bbox[3]/2, bbox[0] + bbox[2]/2]) # center
        self.target_sz = torch.Tensor([bbox[3], bbox[2]]) # real size in pixel


        # TrTr
        bbox_xyxy  = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        channel_avg = np.mean(image, axis=(0, 1))
        # get crop
        s_z, scale_z = siamfc_like_scale(bbox_xyxy)
        s_x = self.search_size / scale_z
        template_image, _ = crop_image(image, bbox_xyxy, padding = channel_avg)

        # get mask
        ## reverse order of target_pos and target_sz := [y,x]
        self.init_template_mask = [0, 0, template_image.shape[1], template_image.shape[0]] # [x1, y1, x2, y2]
        if self.target_pos[1] < s_z/2: # x1
            self.init_template_mask[0] = (s_z/2 - self.target_pos[1]) * scale_z
        if self.target_pos[0] < s_z/2: # y1
            self.init_template_mask[1] = (s_z/2 - self.target_pos[0]) * scale_z
        if self.target_pos[1] + s_z/2 > image.shape[0]: # x2
            self.init_template_mask[2] = self.init_template_mask[2] - (self.target_pos[1] + s_z/2 - image.shape[1]) * scale_z
        if self.target_pos[1] + s_z/2 > image.shape[1]: # y2
            self.init_template_mask[3] = self.init_template_mask[3] - (self.target_pos[0] + s_z/2 - image.shape[0]) * scale_z
        # normalize and conver to torch.tensor
        self.init_template = self.image_normalize(np.round(template_image).astype(np.uint8)).cuda()
        self.first_frame = True


        # Initialize some stuff
        self.atom_frame_num = 1
        if not self.atom_params.has('device'):
            self.atom_params.device = 'cuda' if self.atom_params.use_gpu else 'cpu'

        # Initialize features
        self.atom_params.features.initialize()

        # Set search area
        self.atom_img_sample_sz = self.search_size * torch.ones(2)
        self.atom_target_scale = 1 / scale_z
        self.atom_base_target_sz = self.target_sz / self.atom_target_scale

        # Set sizes
        self.atom_img_support_sz = self.atom_img_sample_sz
        self.atom_feature_sz = self.atom_params.features.size(self.atom_img_sample_sz)
        self.atom_output_sz = self.atom_img_support_sz # use fourier to get same size with img_support_sz
        self.atom_kernel_size = self.atom_fparams.attribute('kernel_size')
        #print(self.atom_img_support_sz, self.atom_feature_sz, self.atom_output_sz, self.atom_kernel_size)

        # Optimization options
        self.atom_output_window = dcf.hann2d(self.atom_output_sz.long(), centered=False).to(self.atom_params.device)

        # Initialize some learning things
        self.atom_init_learning()

        # Setup scale bounds
        self.atom_image_sz = torch.Tensor([image.shape[0], image.shape[1]]) # height, width

        # Extract and transform sample
        x = self.atom_generate_init_samples(image, channel_avg)

        # Initialize projection matrix
        self.atom_init_projection_matrix(x)

        # Transform to get the training sample
        train_x = x

        # Generate label function
        init_y = self.atom_init_label_function(train_x)

        # Init memory
        self.atom_init_memory(train_x)

        # Init optimizer and do initial optimization
        start = time.time()
        self.atom_init_optimization(train_x, init_y)
        #print("init optimization time: {}".format(time.time() - start))

    def _bbox_clip(self, bbox, boundary):
        x1 = max(0, bbox[0])
        y1 = max(0, bbox[1])
        x2 = min(boundary[1], bbox[2])
        y2 = min(boundary[0], bbox[3])

        return [x1, y1, x2, y2]

    def atom_init_optimization(self, train_x, init_y):
        # Initialize filter
        filter_init_method = self.atom_params.get('filter_init_method', 'zeros')
        self.atom_filter = TensorList(
            [x.new_zeros(1, cdim, sz[0], sz[1]) for x, cdim, sz in zip(train_x, self.atom_compressed_dim, self.atom_kernel_size)])
        if filter_init_method == 'zeros':
            pass
        elif filter_init_method == 'randn':
            for f in self.atom_filter:
                f.normal_(0, 1/f.numel())
        else:
            raise ValueError('Unknown "filter_init_method"')

        # Setup factorized joint optimization
        self.atom_joint_problem = FactorizedConvProblem(self.atom_init_training_samples, init_y, self.atom_filter_reg,
                                                   self.atom_fparams.attribute('projection_reg'), self.atom_params, self.atom_init_sample_weights,
                                                   self.atom_projection_activation, self.atom_response_activation)

        # Variable containing both filter and projection matrix
        joint_var = self.atom_filter.concat(self.atom_projection_matrix)

        # Initialize optimizer
        self.atom_joint_optimizer = GaussNewtonCG(self.atom_joint_problem, joint_var)

        # Do joint optimization
        self.atom_joint_optimizer.run(self.atom_params.init_CG_iter // self.atom_params.init_GN_iter, self.atom_params.init_GN_iter)


        # Re-project samples with the new projection matrix
        compressed_samples = self.atom_project_sample(self.atom_init_training_samples, self.atom_projection_matrix)
        for train_samp, init_samp in zip(self.atom_training_samples, compressed_samples):
            train_samp[:init_samp.shape[0],...] = init_samp


        # Initialize optimizer
        self.atom_conv_problem = ConvProblem(self.atom_training_samples, self.atom_y, self.atom_filter_reg, self.atom_sample_weights, self.atom_response_activation)
        self.atom_filter_optimizer = ConjugateGradient(self.atom_conv_problem, self.atom_filter, fletcher_reeves=self.atom_params.fletcher_reeves)

        # Transfer losses from previous optimization
        self.atom_filter_optimizer.residuals = self.atom_joint_optimizer.residuals
        self.atom_filter_optimizer.losses = self.atom_joint_optimizer.losses

        # Post optimization
        self.atom_filter_optimizer.run(self.atom_params.post_init_CG_iter)

        # Free memory
        del self.atom_init_training_samples
        del self.atom_joint_problem, self.atom_joint_optimizer

    def track(self, image):

        self.atom_frame_num += 1

        # get padding value
        channel_avg = np.mean(image, axis=(0, 1))

        # ------- online heatmap and localization ------- #

        # Get sample
        prev_pos = self.target_pos
        sample_pos = self.target_pos.round()
        sample_scales = self.atom_target_scale * self.atom_params.scale_factors
        test_x = self.atom_extract_processed_sample(image, channel_avg, self.target_pos, sample_scales, self.atom_img_sample_sz)

        # Compute scores
        scores_raw = self.atom_apply_filter(test_x)
        translation_vec, scale_ind, s, flag = self.atom_localize_target(scores_raw)
        atom_heatmap = torch.clamp(s[0], min = 0)
        atom_max_score = torch.max(atom_heatmap).item()


        if flag != 'not_found':
            # Update pos
            new_pos = sample_pos + translation_vec

            # boundary margin
            inside_ratio = 0.2
            inside_offset = (inside_ratio - 0.5) * self.target_sz

            self.target_pos = torch.max(torch.min(new_pos, self.atom_image_sz - inside_offset), inside_offset)
        else:
            atom_heatmap = None
            raise ValueError("we should not get not_found result from online heatmap updating")

        # Combine with TrTr tracking framework
        out = self.trtr_tracking(image, channel_avg, prev_pos[[1,0]], self.target_sz[[1,0]], atom_heatmap)


        # ------- online heatmap ------- #

        # Check flags and set learning rate if hard negative
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.atom_params.hard_negative_learning_rate if hard_negative else None

        if update_flag:
            # Get train sample
            train_x = TensorList([x[scale_ind:scale_ind+1, ...] for x in test_x])

            # Create label for sample
            train_y = self.atom_get_label_function(sample_pos, sample_scales[scale_ind])

            # Update memory
            self.atom_update_memory(train_x, train_y, learning_rate)

        # Train filter
        if hard_negative:
            self.atom_filter_optimizer.run(self.atom_params.hard_negative_CG_iter)
        elif (self.atom_frame_num-1) % self.atom_params.train_skipping == 0:
            start = time.time()
            self.atom_filter_optimizer.run(self.atom_params.CG_iter)
            #print("online updating time: {}".format(time.time() - start))

        # Return new state

        # NOTE: if you use a tensor and [[1:0]], which got worse performance, don't know why
        new_state = [self.target_pos[1] - self.target_sz[1] / 2, self.target_pos[0] - self.target_sz[0] / 2,
                     self.target_pos[1] + self.target_sz[1] / 2, self.target_pos[0] + self.target_sz[0] / 2]

        out['bbox'] =  new_state

        out['score'] = atom_max_score

        out['atom_heatmap'] = (torch.round(atom_heatmap.permute(1,2,0) * 255)).detach().cpu().numpy().astype(np.uint8)
        if 'resized_atom_heatmap' in out:
            if out['resized_atom_heatmap'] is not None:
                out['atom_heatmap'] = (np.round(out['resized_atom_heatmap'] * 255)).astype(np.uint8)

        return out

    def trtr_tracking(self, img, channel_avg, prev_pos, prev_sz, atom_heatmap = None):

        prev_bbox_xyxy = [prev_pos[0] - prev_sz[0] / 2,
                          prev_pos[1] - prev_sz[1] / 2,
                          prev_pos[0] + prev_sz[0] / 2,
                          prev_pos[1] + prev_sz[1] / 2]
        _, search_image = crop_image(img, prev_bbox_xyxy, padding = channel_avg, instance_size = self.search_size)
        s_z, scale_z = siamfc_like_scale(prev_bbox_xyxy)
        s_x = self.search_size / scale_z

        if atom_heatmap is not None:
            atom_real_search_size = (self.atom_img_sample_sz * self.atom_target_scale)[0]
            trtr_real_search_size = s_x
            trtr_atom_scale = trtr_real_search_size / atom_real_search_size
            resized_bbox = torch.cat([(1 - trtr_atom_scale) * self.atom_img_sample_sz / 2, (1 + trtr_atom_scale) * self.atom_img_sample_sz / 2])
            resized_atom_heatmap =  crop_hwc(atom_heatmap.permute(1,2,0).detach().cpu().numpy(), resized_bbox, self.heatmap_size)
            #print(resized_bbox, resized_atom_heatmap.shape)
            unroll_resized_atom_heatmap = torch.tensor(resized_atom_heatmap).view(self.heatmap_size * self.heatmap_size)
            best_idx = torch.argmax(unroll_resized_atom_heatmap)
            #print("the peak in atom hetmap: {}".format([best_idx % self.heatmap_size, best_idx // self.heatmap_size]))
        else:
            resized_atom_heatmap = None

        # get mask
        search_mask = [0, 0, search_image.shape[1], search_image.shape[0]]
        if prev_pos[0] < s_x/2:
            search_mask[0] = (s_x/2 - prev_pos[0]) * scale_z
        if prev_pos[1] < s_x/2:
            search_mask[1] = (s_x/2 - prev_pos[1]) * scale_z
        if prev_pos[0] + s_x/2 > img.shape[1]:
            search_mask[2] = search_mask[2] - (prev_pos[0] + s_x/2 - img.shape[1]) * scale_z
        if prev_pos[1] + s_x/2 > img.shape[0]:
            search_mask[3] = search_mask[3] - (prev_pos[1] + s_x/2 - img.shape[0]) * scale_z

        # normalize and conver to torch.tensor
        search = self.image_normalize(np.round(search_image).astype(np.uint8)).cuda()

        with torch.no_grad():
            if self.first_frame:
                outputs = self.model(nested_tensor_from_tensor_list([search], [torch.as_tensor(search_mask).float()]),
                                     nested_tensor_from_tensor_list([self.init_template], [torch.as_tensor(self.init_template_mask).float()]))
                self.first_frame = False
            else:
                outputs = self.model(nested_tensor_from_tensor_list([search], [torch.as_tensor(search_mask).float()]))

        outputs = self.postprocess(outputs)

        heatmap = outputs['pred_heatmap'][0].cpu() # we only address with a single image
        raw_heatmap = heatmap.view(self.heatmap_size, self.heatmap_size) # as a image format
        found = torch.max(heatmap) > self.score_threshold

        if atom_heatmap is not None:
            found = True

        if not found:
            return {}

        def change(r):
            return torch.max(r, 1. / r)

        bbox_wh_map = outputs['pred_bbox_wh'][0].cpu() * torch.as_tensor(search.shape[-2:])  # convert from relative [0, 1] to absolute [0, height] coordinates

        # scale penalty
        pad = (bbox_wh_map[:, 0] + bbox_wh_map[:, 1]) * get_context_amount()
        sz = torch.sqrt((bbox_wh_map[:, 0] + pad) * (bbox_wh_map[:, 1] + pad))
        s_c = change(sz / get_exemplar_size())

        # aspect ratio penalty
        r_c = change((bbox_wh_map[:, 0] / bbox_wh_map[:, 1]) / (prev_sz[0] / prev_sz[1]) )
        penalty = torch.exp(-(r_c * s_c - 1) * self.size_penalty_k)

        best_idx = 0
        window_factor = self.window_factor
        post_heatmap = None
        best_score = 0

        for i in range(self.window_steps):
            # add distance penalty
            post_heatmap = penalty * heatmap * (1 -  window_factor) + self.window * window_factor
            best_idx = torch.argmax(post_heatmap)
            best_score = heatmap[best_idx].item()

            if best_score > self.score_threshold:
                break;
            else:
                window_factor = np.max(window_factor - self.window_factor / self.window_steps, 0)

        if atom_heatmap is not None:
            atom_rate = 0.5 # heuristic
            post_heatmap = post_heatmap * (1 -  atom_rate) + unroll_resized_atom_heatmap * atom_rate
            best_idx = torch.argmax(post_heatmap)
            best_score = post_heatmap[best_idx].item()

        post_heatmap = post_heatmap.view(self.heatmap_size, self.heatmap_size) # as a image format

        # bbox
        ct_int = torch.stack([best_idx % self.heatmap_size, best_idx // self.heatmap_size], dim = -1)
        #print("the peak in trtr hetmap: {}".format([best_idx % self.heatmap_size, best_idx // self.heatmap_size]))
        bbox_reg = outputs['pred_bbox_reg'][0][best_idx].cpu()
        bbox_ct = (ct_int + bbox_reg) * torch.as_tensor(search.shape[-2:]) / float(self.heatmap_size)
        bbox_wh = bbox_wh_map[best_idx]

        ct_delta = (bbox_ct - self.search_size / 2) / scale_z
        cx = prev_pos[0] + ct_delta[0].item()
        cy = prev_pos[1] + ct_delta[1].item()

        # smooth bbox
        lpf = best_score * self.size_lpf
        bbox_wh = bbox_wh / scale_z

        width = prev_sz[0] * (1 - lpf) + bbox_wh[0].item() * lpf
        height = prev_sz[1] * (1 - lpf) + bbox_wh[1].item() * lpf


        # clip boundary
        bbox = [cx - width / 2, cy - height / 2,
                cx + width / 2, cy + height / 2]
        bbox = self._bbox_clip(bbox, img.shape[:2])

        # udpate state ([y,x])
        self.target_pos = torch.Tensor([(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2])
        self.target_sz = torch.Tensor([bbox[3] - bbox[1], bbox[2] - bbox[0]])


        # debug for search image:
        debug_bbox = torch.round(box_cxcywh_to_xyxy(torch.cat([bbox_ct, bbox_wh * scale_z]))).int()
        rec_search_image = cv2.rectangle(search_image,
                                         (debug_bbox[0], debug_bbox[1]),
                                         (debug_bbox[2], debug_bbox[3]),(0,255,0),3)

        raw_heatmap = (torch.round(raw_heatmap * 255)).detach().numpy().astype(np.uint8)
        post_heatmap = (torch.round(post_heatmap * 255)).detach().numpy().astype(np.uint8)
        heatmap_resize = cv2.resize(raw_heatmap, search_image.shape[1::-1])
        heatmap_color = np.stack([heatmap_resize, np.zeros(search_image.shape[1::-1], dtype=np.uint8), heatmap_resize], -1)
        rec_search_image = np.round(0.4 * heatmap_color + 0.6 * rec_search_image.copy()).astype(np.uint8)


        # self.atom_target_scale = torch.sqrt(self.target_sz.prod() / self.atom_base_target_sz.prod())
        self.atom_target_scale = 1 / siamfc_like_scale(bbox)[1] # TODO: redundant with the begining of this functin

        return {
            'bbox': bbox,
            'score': best_score,
            'raw_heatmap': raw_heatmap,
            'post_heatmap': post_heatmap,
            'search_image': rec_search_image, # debug
            'resized_atom_heatmap': resized_atom_heatmap,
        }


    def atom_apply_filter(self, sample_x: TensorList):
        return operation.conv2d(sample_x, self.atom_filter, mode='same')

    def atom_localize_target(self, scores_raw):
        # Weighted sum (if multiple features) with interpolation in fourier domain
        weight = self.atom_fparams.attribute('translation_weight', 1.0)
        scores_raw = weight * scores_raw
        # TODO: learn the mechanism
        sf_weighted = fourier.cfft2(scores_raw) / (scores_raw.size(2) * scores_raw.size(3))
        for i, (sz, ksz) in enumerate(zip(self.atom_feature_sz, self.atom_kernel_size)):
            sf_weighted[i] = fourier.shift_fs(sf_weighted[i], math.pi * (1 - torch.Tensor([ksz[0]%2, ksz[1]%2]) / sz))

        scores_fs = fourier.sum_fs(sf_weighted)
        scores = fourier.sample_fs(scores_fs, self.atom_output_sz)

        scores *= self.atom_output_window

        sz = scores.shape[-2:]

        # Shift scores back
        scores = torch.cat([scores[...,(sz[0]+1)//2:,:], scores[...,:(sz[0]+1)//2,:]], -2)
        scores = torch.cat([scores[...,:,(sz[1]+1)//2:], scores[...,:,:(sz[1]+1)//2]], -1)


        # Find maximum
        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - self.atom_output_sz // 2
        translation_vec1 = target_disp1 * (self.atom_img_support_sz / self.atom_output_sz) * self.atom_target_scale

        if max_score1.item() < self.atom_params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores, 'not_found'

        # Mask out target neighborhood
        target_neigh_sz = self.atom_params.target_neighborhood_scale * self.target_sz / self.atom_target_scale
        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores[scale_ind:scale_ind+1,...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - self.atom_output_sz // 2
        translation_vec2 = target_disp2 * (self.atom_img_support_sz / self.atom_output_sz) * self.atom_target_scale

        # Handle the different cases
        if max_score2 > self.atom_params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum(target_disp1**2))
            disp_norm2 = torch.sqrt(torch.sum(target_disp2**2))
            disp_threshold = self.atom_params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores, 'uncertain'

        if max_score2 > self.atom_params.hard_negative_threshold * max_score1 and max_score2 > self.atom_params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores, 'hard_negative'

        return translation_vec1, scale_ind, scores, None


    def atom_extract_processed_sample(self, im: np.ndarray, padding_value: float, pos: torch.Tensor, scales, sz: torch.Tensor) -> (TensorList, TensorList):
        x = self.atom_params.features.extract(im, pos, scales, sz, padding_value)
        return self.atom_project_sample(x)


    def atom_project_sample(self, x: TensorList, proj_matrix = None):
        # Apply projection matrix
        if proj_matrix is None:
            proj_matrix = self.atom_projection_matrix
        return operation.conv2d(x, proj_matrix).apply(self.atom_projection_activation)

    def atom_init_learning(self):
        # Get window function
        self.atom_feature_window = TensorList([dcf.hann2d(sz).to(self.atom_params.device) for sz in self.atom_feature_sz])

        # Filter regularization
        self.atom_filter_reg = self.atom_fparams.attribute('filter_reg')

        # Activation function after the projection matrix (phi_1 in the paper)
        projection_activation = self.atom_params.get('projection_activation', 'none')
        if isinstance(projection_activation, tuple):
            projection_activation, act_param = projection_activation

        if projection_activation == 'none':
            self.atom_projection_activation = lambda x: x
        elif projection_activation == 'relu':
            self.atom_projection_activation = torch.nn.ReLU(inplace=True)
        elif projection_activation == 'elu':
            self.atom_projection_activation = torch.nn.ELU(inplace=True)
        elif projection_activation == 'mlu':
            self.atom_projection_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')

        # Activation function after the output scores (phi_2 in the paper)
        response_activation = self.atom_params.get('response_activation', 'none')
        if isinstance(response_activation, tuple):
            response_activation, act_param = response_activation

        if response_activation == 'none':
            self.atom_response_activation = lambda x: x
        elif response_activation == 'relu':
            self.atom_response_activation = torch.nn.ReLU(inplace=True)
        elif response_activation == 'elu':
            self.atom_response_activation = torch.nn.ELU(inplace=True)
        elif response_activation == 'mlu':
            self.atom_response_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')


    def atom_generate_init_samples(self, im: np.ndarray, padding_value: float):
        """Generate augmented initial samples."""

        # Compute augmentation size
        aug_expansion_factor = self.atom_params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.atom_img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.atom_img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.atom_img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.atom_img_sample_sz.long().tolist()

        # Random shift operator
        get_rand_shift = lambda: None
        random_shift_factor = self.atom_params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.atom_img_sample_sz * random_shift_factor).long().tolist()

        # Create transformations
        self.atom_transforms = [augmentation.Identity(aug_output_sz)]
        if 'shift' in self.atom_params.augmentation:
            self.atom_transforms.extend([augmentation.Translation(shift, aug_output_sz) for shift in self.atom_params.augmentation['shift']])
        if 'relativeshift' in self.atom_params.augmentation:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.atom_img_sample_sz/2).long().tolist()
            self.atom_transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz) for shift in self.atom_params.augmentation['relativeshift']])
        if 'fliplr' in self.atom_params.augmentation and self.atom_params.augmentation['fliplr']:
            self.atom_transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in self.atom_params.augmentation:
            self.atom_transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in self.atom_params.augmentation['blur']])
        if 'scale' in self.atom_params.augmentation:
            self.atom_transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in self.atom_params.augmentation['scale']])
        if 'rotate' in self.atom_params.augmentation:
            self.atom_transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in self.atom_params.augmentation['rotate']])

        # Generate initial samples
        init_samples = self.atom_params.features.extract_transformed(im, self.target_pos, self.atom_target_scale, aug_expansion_sz, self.atom_transforms, padding_value)

        # Remove augmented samples for those that shall not have
        for i, use_aug in enumerate(self.atom_fparams.attribute('use_augmentation')):
            if not use_aug:
                init_samples[i] = init_samples[i][0:1, ...]

        #print("====     atom_init_samples: ", init_samples[0].shape)
        # Add dropout samples
        if 'dropout' in self.atom_params.augmentation:
            num, prob = self.atom_params.augmentation['dropout']
            self.atom_transforms.extend(self.atom_transforms[:1]*num)
            for i, use_aug in enumerate(self.atom_fparams.attribute('use_augmentation')):
                if use_aug:
                    init_samples[i] = torch.cat([init_samples[i], F.dropout2d(init_samples[i][0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        return init_samples


    def atom_init_projection_matrix(self, x):

        self.atom_compressed_dim = self.atom_fparams.attribute('compressed_dim', None)
        self.atom_projection_matrix = TensorList(
            [None if cdim is None else ex.new_zeros(cdim,ex.shape[1],1,1).normal_(0,1/math.sqrt(ex.shape[1])) for ex, cdim in
             zip(x, self.atom_compressed_dim)])
        #print("====     atom_projection_matrix: ", self.atom_projection_matrix[0].shape)

    def atom_init_label_function(self, train_x):
        # Allocate label function
        self.atom_y = TensorList([x.new_zeros(self.atom_params.sample_memory_size, 1, x.shape[2], x.shape[3]) for x in train_x])

        # Output sigma factor
        output_sigma_factor = self.atom_fparams.attribute('output_sigma_factor')
        self.atom_sigma = (self.atom_feature_sz / self.atom_img_support_sz * self.atom_base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(2)

        # Center pos in normalized coords (offset becuase of the float)
        target_center_norm = (self.target_pos - self.target_pos.round()) / (self.atom_target_scale * self.atom_img_support_sz)

        # Generate label functions
        for y, sig, sz, ksz, x in zip(self.atom_y, self.atom_sigma, self.atom_feature_sz, self.atom_kernel_size, train_x):
            center_pos = sz * target_center_norm + 0.5 * torch.Tensor([(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
            for i, T in enumerate(self.atom_transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(T.shift) / self.atom_img_support_sz * sz
                y[i, 0, ...] = dcf.label_function_spatial(sz, sig, sample_center)

        # Return only the ones to use for initial training
        return TensorList([y[:x.shape[0], ...] for y, x in zip(self.atom_y, train_x)])


    def atom_init_memory(self, train_x):
        # Initialize first-frame training samples
        self.atom_num_init_samples = train_x.size(0)
        self.atom_init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])
        self.atom_init_training_samples = train_x

        # Sample counters and weights
        self.atom_num_stored_samples = self.atom_num_init_samples.copy()
        self.atom_previous_replace_ind = [None] * len(self.atom_num_stored_samples)
        self.atom_sample_weights = TensorList([x.new_zeros(self.atom_params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.atom_sample_weights, self.atom_init_sample_weights, self.atom_num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.atom_training_samples = TensorList(
            [x.new_zeros(self.atom_params.sample_memory_size, cdim, x.shape[2], x.shape[3]) for x, cdim in
             zip(train_x, self.atom_compressed_dim)])

    def atom_update_memory(self, sample_x: TensorList, sample_y: TensorList, learning_rate = None):
        replace_ind = self.atom_update_sample_weights(self.atom_sample_weights, self.atom_previous_replace_ind, self.atom_num_stored_samples, self.atom_num_init_samples, self.atom_fparams, learning_rate)
        self.atom_previous_replace_ind = replace_ind
        for train_samp, x, ind in zip(self.atom_training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x
        for y_memory, y, ind in zip(self.atom_y, sample_y, replace_ind):
            y_memory[ind:ind+1,...] = y
        self.atom_num_stored_samples += 1


    def atom_update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, fparams, learning_rate = None):
        # Update weights and get index to replace in memory
        replace_ind = []
        for sw, prev_ind, num_samp, num_init, fpar in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, fparams):
            lr = learning_rate
            if lr is None:
                lr = fpar.learning_rate

            init_samp_weight = getattr(fpar, 'init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                _, r_ind = torch.min(sw[s_ind:], 0)
                r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def atom_get_label_function(self, sample_pos, sample_scale):
        # Generate label function
        train_y = TensorList()
        target_center_norm = (self.target_pos - sample_pos) / (sample_scale * self.atom_img_support_sz)
        for sig, sz, ksz in zip(self.atom_sigma, self.atom_feature_sz, self.atom_kernel_size):
            center = sz * target_center_norm + 0.5 * torch.Tensor([(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
            train_y.append(dcf.label_function_spatial(sz, sig, center))
        return train_y

def build_tracker(args):

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available in Pytorch")

    device = torch.device('cuda')

    assert args.transformer_mask # should be True
    model, _, postprocessors = build_model(args)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    assert 'model' in checkpoint
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    return Tracker(model, postprocessors["bbox"], args.search_size, args.window_factor, args.score_threshold, args.window_steps, args.tracking_size_penalty_k, args.tracking_size_lpf)
