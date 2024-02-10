_base_ = ['../base/dataset_hubmap.py']

test_param = dict(
    model_param = dict(
        output_exp = dict(sclerosis = "./work_dir/sclerosis",
                          glomerulus = "./work_dir/glomerulus"),
        filename_checkpoint = "epoch_50.pth",
        encoder_name = "efficientnet-b0",
        encoder_weights = "imagenet",
        classes = 1,
        activation = None,
        network_name = "unet",
    ),
    model = "unet",
    input_resolution = 320,
    resolution = 1024,
    pad_size = 0,
    clf_threshold = 0.5,
    small_mask_threshold = 0,
    mask_threshold = 0.5,
    tta = 3,
    test_batch_size = 12,
    num_workers = 4,
) 
