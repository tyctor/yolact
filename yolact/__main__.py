from . import Yolact
from .data import config
import argparse
import torch

# unofficial yolact to onnx script
parser = argparse.ArgumentParser(
    description='Yolact Convert to onnx script')
parser.add_argument('--weights', default=None, type=str,
                    help='Weights')
parser.add_argument('--shape', default=None, type=tuple,
                    help='Input shape')
parser.add_argument('--config', default=None, type=dict,
                    help='config to use')

parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
args = parser.parse_args()

test_dataset = config.Config({
    'name': 'Test',

    # Training images and annotations
    'train_images': '',
    'train_info':   '',

    # Validation images and annotations.
    'valid_images': '',
    'valid_info':   '',

    # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
    'has_gt': '',

    # A list of names for each of you classes.
    'class_names': ('err',),

    # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
    # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
    # If not specified, this just assumes category ids start at 1 and increase sequentially.
    'label_map': None
})

test_config = config.coco_base_config.copy({
    'name': 'test_base',
    'typ':'normal',
    # Dataset stuff
    'dataset': test_dataset,
    'num_classes': len(test_dataset.class_names) + 1,
    # Image Size
    'max_size': 550,
    # Training params
    'lr_warmup_init': 1e-6,
    'lr_steps': (280000, 600000, 700000, 750000),
    'max_iter': 800000,
    # Backbone Settings
    'backbone': config.resnet50_backbone.copy({
        'selected_layers': list(range(1, 4)),
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': True, # This is for backward compatability with a bug
        'pred_aspect_ratios': [ [[1, 1/2, 2]] ]*5,
        'pred_scales': [[24], [48], [96], [192], [384]],
    }),
    # FPN Settings
    'fpn': config.fpn_base.copy({
        'use_conv_downsample': True,
        'num_downsample': 2,
    }),
    # Mask Settings
    'mask_type': config.mask_type.lincomb,
    'mask_alpha': 6.125,
    'mask_proto_src': 0,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})],
    'mask_proto_normalize_emulate_roi_pooling': True,
    # SSD data augmentation parameters
    # Randomize hue, vibrance, etc.
    'augment_photometric_distort': False,
    # Have a chance to scale down the image and pad (to emulate smaller detections)
    'augment_expand': False,
    # Potentialy sample a random crop from the image and put it in a random place
    'augment_random_sample_crop': False,
    # Mirror the image with a probability of 1/2
    'augment_random_mirror': True,
    # Flip the image vertically with a probability of 1/2
    'augment_random_flip': True,
    # With uniform probability, rotate the image [0,90,180,270] degrees
    'augment_random_rot90': True,
    # Other stuff
    'display_lincomb' : False,
    # FROM args TO cfg for eval
    'crop' : True,
    'score_threshold' : 0.1,
    'top_k' : 300,
    'display_masks' : True,
    'display_fps' : False,
    'display_text' : True,
    'display_bboxes' : True,
    'display_scores' : True,
    'bbox_det_file' : 'results/bbox_detections.json',
    'mask_det_file' : 'results/mask_detections.json',
    'fast_nms' : True,
    'cross_class_nms' : False,
    'mask_proto_debug' : False,
    'share_prediction_module': True,
    'extra_head_net': [(256, 3, {'padding': 1})],
    'positive_iou_threshold': 0.5,
    'negative_iou_threshold': 0.4,
    'crowd_iou_threshold': 0.7,
    'use_semantic_segmentation_loss': True,
})
def say(msg: str):
    print(f"\n{msg}\n")
# stupid onnx convert addon
def convert_to_onnx(weights, input_shape):
    try: 
        say(f"setting config")
        config.set_cfg(test_config)
        
        say(f"creating net")
        net = Yolact()

        say(f"loading weights")
        net.load_weights(weights)
        print(net)

        dummy_input = torch.rand(1, 3, 550, 550)

        say("try to convert")
        torch.onnx.export(net, dummy_input, './yolact.onnx', verbose=True)
    except Exception as e:
        print(f"Exception: {str(e)}")

convert_to_onnx(args.weights, args.shape)