"""
Various positional encodings for the transformer (numpy version)
"""
import math
import numpy as np

class PositionEmbeddingSine():
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, image_size = 255, stride = 8):
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.segment_embdded_factor = 0.0  #0.5, 0.0 gives best result for VOT2018. TODO: hyperparameter

        self.image_size = image_size
        self.feature_size = (image_size + 1) // stride

        mask = np.ones((self.feature_size, self.feature_size))
        self.default_position_embedding = self._algorithm(mask)

    def create(self, mask_bounds):

        mask_bounds = np.array(mask_bounds) * self.feature_size / self.image_size

        if mask_bounds[0] >= 1 or mask_bounds[1] >= 1 or mask_bounds[2] < self.feature_size - 1 or mask_bounds[3] < self.feature_size - 1:
            # create mask
            mask = np.zeros((self.feature_size, self.feature_size))
            mask[math.ceil(mask_bounds[1]): math.floor(mask_bounds[3]), math.ceil(mask_bounds[0]): math.floor(mask_bounds[2])] = 1
            return self._algorithm(mask)
        else:
            return self.default_position_embedding


    def _algorithm(self, mask):

        # Note: can not use different model between training and inference
        y_embed = mask.cumsum(0, dtype=np.float32)
        x_embed = mask.cumsum(1, dtype=np.float32)

        #print("x_embed: {}".format(x_embed.shape))
        #print("x_embed[:, :, -1:]: {}".format(x_embed[:, :, -1:].shape))
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = np.arange(self.num_pos_feats, dtype=np.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t

        #print(np.stack((np.sin(pos_x[:, :, 0::2]), np.cos(pos_x[:, :, 1::2])), 3).shape)
        pos_x = np.stack((np.sin(pos_x[:, :, 0::2]), np.cos(pos_x[:, :, 1::2])), 3).reshape(mask.shape[0], mask.shape[1], -1)
        pos_y = np.stack((np.sin(pos_y[:, :, 0::2]), np.cos(pos_y[:, :, 1::2])), 3).reshape(mask.shape[0], mask.shape[1], -1)

        pos = np.concatenate((pos_y, pos_x), 2).transpose(2, 0, 1)
        return pos

def build_position_encoding(args, image_size):
    N_steps = args.transformer.hidden_dim // 2
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True, image_size = image_size)
    return position_embedding
