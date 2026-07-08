from train_imagenet1k_timm_baseline_common import main_for_model


MODEL_SPEC = {
    "display_name": "EfficientNet-B3",
    "model_name": "efficientnet_b3.in1k",
    "search_pattern": "*efficientnet_b3*",
    "candidates": [
        "efficientnet_b3.in1k",
        "efficientnet_b3.ra2_in1k",
        "tf_efficientnet_b3.in1k",
        "tf_efficientnet_b3",
    ],
    "checkpoint_dir": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\efficientnet_b3",
}


if __name__ == "__main__":
    main_for_model(MODEL_SPEC)
