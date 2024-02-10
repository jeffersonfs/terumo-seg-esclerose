dataset_name = "sclerosis"
path = "./dist/datasets/sclerosis/"
cache = True
batch_size = 8
multiplier_bin = 4
binned_max = 20
split = "train"
dataset_pre_processing = dict(shift_list=[0, 512],tile_size=1024)

train_param = dict(type="train_param",
    output_path = './dist/datasets/sclerosis/',
    data_csv_path = './dist/datasets/sclerosis/patchs/data_balanced.csv',
    test_size = 0.05,
    random_state = 19,
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225],
    batch_size = 12,
    shuffle = True,
    encoder_name = "efficientnet-b0",
    encoder_weights = "imagenet",
    classes = 1,
    activation = None,
    network_name = "unet",
    loop_param = dict(type='loop_param',
        filename_checkpoint = "best_checkpoint.pth",
        cache_weight = None,
        result_csv = "history.csv",
        max_lr = 1e-3,
        epochs = 50,
        weight_decay = 1e-5,
        criterion_name = "bce"),
    transform_param = dict(type='transform_param', img_size = 320)
    )

