import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from time import perf_counter
from sold2.model.line_matcher import LineMatcher
from sold2.misc.visualize_util import plot_images, plot_lines, plot_line_matches, plot_color_line_matches, plot_keypoints

#%% md

# Matching from scratch given pairs of images

#%%

ckpt_path = '../pretrained_models/sold2_wireframe.tar'
device = 'cuda'
mode = 'dynamic'  # 'dynamic' or 'static'

# Initialize the line matcher
config = {
    'model_cfg': {
        'model_name': "lcnn_simple",
        'model_architecture': "simple",
        # Backbone related config
        'backbone': "lcnn",
        'backbone_cfg': {
            'input_channel': 1, # Use RGB images or grayscale images.
            'depth': 4,
            'num_stacks': 2,
            'num_blocks': 1,
            'num_classes': 5
        },
        # Junction decoder related config
        'junction_decoder': "superpoint_decoder",
        'junc_decoder_cfg': {},
        # Heatmap decoder related config
        'heatmap_decoder': "pixel_shuffle",
        'heatmap_decoder_cfg': {},
        # Descriptor decoder related config
        'descriptor_decoder': "superpoint_descriptor",
        'descriptor_decoder_cfg': {},
        # Shared configurations
        'grid_size': 8,
        'keep_border_valid': True,
        # Threshold of junction detection
        'detection_thresh': 0.0153846, # 1/65
        'max_num_junctions': 300,
        # Threshold of heatmap detection
        'prob_thresh': 0.5,
        # Weighting related parameters
        'weighting_policy': mode,
        # [Heatmap loss]
        'w_heatmap': 0.,
        'w_heatmap_class': 1,
        'heatmap_loss_func': "cross_entropy",
        'heatmap_loss_cfg': {
            'policy': mode
        },
        # [Heatmap consistency loss]
        # [Junction loss]
        'w_junc': 0.,
        'junction_loss_func': "superpoint",
        'junction_loss_cfg': {
            'policy': mode
        },
        # [Descriptor loss]
        'w_desc': 0.,
        'descriptor_loss_func': "regular_sampling",
        'descriptor_loss_cfg': {
            'dist_threshold': 8,
            'grid_size': 4,
            'margin': 1,
            'policy': mode
        },
    },
    'line_detector_cfg': {
        'detect_thresh': 0.25,  # depending on your images, you might need to tune this parameter
        'num_samples': 64,
        'sampling_method': "local_max",
        'inlier_thresh': 0.9,
        "use_candidate_suppression": True,
        "nms_dist_tolerance": 3.,
        "use_heatmap_refinement": True,
        "heatmap_refine_cfg": {
            "mode": "local",
            "ratio": 0.2,
            "valid_thresh": 1e-3,
            "num_blocks": 20,
            "overlap_ratio": 0.5
        }
    },
    'multiscale': False,
    'line_matcher_cfg': {
        'cross_check': True,
        'num_samples': 5,
        'min_dist_pts': 8,
        'top_k_candidates': 10,
        'grid_size': 4
    }
}

line_matcher = LineMatcher(
        config["model_cfg"], ckpt_path, device, config["line_detector_cfg"],
        config["line_matcher_cfg"], config["multiscale"])

#%%

# Read and pre-process the images
scale_factor = 2  # we recommend resizing the images to a resolution in the range 400~800 pixels
# img1 = '../assets/images/terrace0.JPG'
img1 = '../assets/images/0000000066.png'
img1 = cv2.imread(img1, 0)
img1 = cv2.resize(img1, (img1.shape[1] // scale_factor, img1.shape[0] // scale_factor),
                  interpolation = cv2.INTER_AREA)
img1 = (img1 / 255.).astype(float)
torch_img1 = torch.tensor(img1, dtype=torch.float)[None, None]
# img2 = '../assets/images/terrace1.JPG'
img2 = '../assets/images/0000000069.png'
img2 = cv2.imread(img2, 0)
img2 = cv2.resize(img2, (img2.shape[1] // scale_factor, img2.shape[0] // scale_factor),
                  interpolation = cv2.INTER_AREA)
img2 = (img2 / 255.).astype(float)
torch_img2 = torch.tensor(img2, dtype=torch.float)[None, None]

# Match the lines
t1 = perf_counter()
outputs = line_matcher([torch_img1, torch_img2])
t2 = perf_counter()
print("time cost: ", t2-t1)
line_seg1 = outputs["line_segments"][0]
line_seg2 = outputs["line_segments"][1]
matches = outputs["matches"]

valid_matches = matches != -1
match_indices = matches[valid_matches]
matched_lines1 = line_seg1[valid_matches][:, :, ::-1]
matched_lines2 = line_seg2[match_indices][:, :, ::-1]

# Plot the matches
plot_images([img1, img2], ['Image 1 - detected lines', 'Image 2 - detected lines'])
plot_lines([line_seg1[:, :, ::-1], line_seg2[:, :, ::-1]], ps=3, lw=2)
plot_images([img1, img2], ['Image 1 - matched lines', 'Image 2 - matched lines'])
plot_color_line_matches([matched_lines1, matched_lines2], lw=2)
